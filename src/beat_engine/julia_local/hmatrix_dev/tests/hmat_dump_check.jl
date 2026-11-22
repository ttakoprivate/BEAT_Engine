# Independent H-matrix solve of a frequency captured with BLAB_HMAT_DUMP (real BLAB project run through the engine).
# usage (julia_local or a work dir with src/ and tests/): julia ... hmat_dump_check.jl <dump.jls> [backend cpu|metal|cuda] [tol]
include(joinpath(pwd(), "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(pwd(), "src", "BeatEngineCoupled.jl"))
include(joinpath(pwd(), "src", "BeatEngineFmm.jl")); using .BeatEngineFmm
using Krylov, LinearMaps, Random, SparseArrays, Printf, StaticArrays, LinearAlgebra, Serialization
const BC = BeatEngineCore
# repository layout: hmatrix_dev/tests/ next to hmatrix_dev/*.jl; Mac kit layout: all files in one folder
const HMAT_DIR = isfile(joinpath(@__DIR__, "hmat_core.jl")) ? (@__DIR__) : joinpath(@__DIR__, "..")
include(joinpath(HMAT_DIR, "hmat_core.jl")); include(joinpath(HMAT_DIR, "hmat_fast.jl"))
include(joinpath(HMAT_DIR, "hmat_ka.jl")); include(joinpath(HMAT_DIR, "hmat_c.jl"))
function hmat_load(dir)
    vals = Pair{Symbol,Any}[]
    rd(name, T, dims) = (a = Array{T}(undef, dims...); read!(joinpath(dir, name * ".bin"), a); a)
    arrays = Dict{String,Any}()
    lines = readlines(joinpath(dir, "manifest.txt"))
    for l in lines
        w = split(l); w[2] == "array" || continue
        T = Dict("Int64" => Int64, "Float64" => Float64, "ComplexF64" => ComplexF64, "Float32" => Float32, "ComplexF32" => ComplexF32)[w[3]]
        arrays[w[1]] = rd(w[1], T, parse.(Int, w[4:end]))
    end
    for l in lines
        w = split(l); name = w[1]; kind = w[2]
        occursin("__", name) && continue
        if kind == "array"; push!(vals, Symbol(name) => arrays[name])
        elseif kind == "sparse"
            push!(vals, Symbol(name) => sparse(arrays[name * "__I"], arrays[name * "__J"], arrays[name * "__V"], parse(Int, w[3]), parse(Int, w[4])))
        elseif kind == "range"; push!(vals, Symbol(name) => parse(Int, w[3]):parse(Int, w[4]))
        elseif kind == "mesh"
            V = arrays[name * "__vertices"]; F = arrays[name * "__faces"]
            push!(vals, Symbol(name) => BoundaryMesh([SVector{3,Float64}(V[:, i]) for i in axes(V, 2)],
                [(F[1, i], F[2, i], F[3, i]) for i in axes(F, 2)], Vector{Int}(arrays[name * "__tags"])))
        elseif kind == "nothing"; push!(vals, Symbol(name) => nothing)
        elseif kind == "scalar"; push!(vals, Symbol(name) => (w[3] in ("Int64",) ? parse(Int, w[4]) : parse(Float64, w[4])))
        else push!(vals, Symbol(name) => Symbol(w[3]))
        end
    end
    return (; vals...)
end
d = isdir(ARGS[1]) ? hmat_load(ARGS[1]) : deserialize(ARGS[1])
gpu = length(ARGS) > 1 ? ARGS[2] : "cpu"; tol = length(ARGS) > 2 ? parse(Float64, ARGS[3]) : 1e-4
todev = identity; Tg = gpu in ("cpu", "cuda64") ? Float64 : Float32
gpu == "metal" && (@eval using Metal; todev = MtlArray); gpu in ("cuda", "cuda64") && (@eval using CUDA; todev = CuArray)
bem = d.bem_mesh; k = Float64(d.wavenumber)
order = d.regular_quadrature_order
p1 = build_p1_space(bem); dp0 = build_dp0_space(bem); rule = triangle_rule(Float64, order)
t0 = @elapsed op = fmm_burton_miller_operator(bem, p1, dp0, k, rule; identity_p1_p1=SparseMatrixCSC{Float64,Int}(real.(d.identity_p1_p1)), identity_p1_dp0=SparseMatrixCSC{Float64,Int}(real.(d.identity_p1_dp0)),
    singular_order=d.singular_order, symmetry_mode=d.symmetry_mode, eps=1e-6)
o = FastOracle(bem, p1, dp0, k, rule, op, d.symmetry_mode); g = ka_geom(o, Tg, todev)
B, Q, Mr, Er, G = collect(d.bem_range), collect(d.flux_range), collect(d.mechanical_range), collect(d.electrical_range), collect(d.gamma_range)
nB = length(B); n = size(d.rest, 1)
@printf("dump %.0f Hz (order %d, singular %d, symmetry %s): n %d = Γ %d + BEM %d + q %d + mech %d + elec %d, rhs columns %d, near setup %.0fs\n",
    d.frequency_hz, order, d.singular_order, d.symmetry_mode, n, length(G), nB, length(Q), length(Mr), length(Er), size(d.rhs, 2), t0)
# entry check against the engine's dense rows
mI = zeros(Int, nB); mJ = zeros(Int, nB); r = 1:size(d.bem_lhs_rows, 1)
E = exact_block_fast(o, collect(r), _faces(o, collect(r)), collect(1:nB), _faces(o, collect(1:nB)), mI, mJ)
ep = norm(E - d.bem_lhs_rows) / norm(d.bem_lhs_rows); ec = norm(conj.(E) - d.bem_lhs_rows) / norm(d.bem_lhs_rows)
@printf("   A rows vs engine dense: rel err %.1e (as is), %.1e (conjugate phasor convention)\n", ep, ec)
if ec < ep
    # the engine's system is the complex conjugate of this oracle's (exp(+i omega t) phasors): solve the conjugate system
    println("   -> engine uses the conjugate phasor convention: solving conj(M) conj(x) = conj(b)")
    d = merge(d, (rest=conj.(d.rest), bem_flux=conj.(d.bem_flux),
        bem_motion_flux=d.bem_motion_flux === nothing ? nothing : conj.(d.bem_motion_flux),
        rhs=conj.(d.rhs), dense_solution=conj.(d.dense_solution), bem_iface_rows=conj.(d.bem_iface_rows)))
end
X = reduce(hcat, bem.vertices); root = hcluster(X, collect(1:nB), 48)
trs = BC.symmetry_image_transforms(BC.normalized_symmetry_mode(d.symmetry_mode))
imgV = [[BC.reflect_point(t, v) for v in bem.vertices] for t in trs]
function adm(a, b, eta)
    min(hdiam(a), hdiam(b)) <= eta * hdist(a, b) || return false
    for iv in imgV
        P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)), vec(maximum(P, dims=2)), HCluster[])
        min(hdiam(a), hdiam(b)) <= eta * hdist(a, bb) || return false
    end
    true
end
ka_reg_entries(g, todev(Int32[1, 2]), todev(Int32[1, 2]))
tA = @elapsed (HA, _, _) = hbuild_gpu(o, g, todev, root, (a, b) -> adm(a, b, 1.0), tol, nB)
flux = d.bem_flux; mflux = d.bem_motion_flux === nothing ? spzeros(ComplexF64, size(flux, 1), 0) : sparse(d.bem_motion_flux)
dofof = [el.dp0_dof for el in op.elements]
used = falses(dp0.global_dof_count); used[unique(rowvals(flux))] .= true; isempty(mflux) || (used[unique(rowvals(mflux))] .= true)
cols = [fc for fc in eachindex(op.elements) if used[dofof[fc]]]; colsdof = dofof[cols]
XF = reduce(hcat, [vec(sum(o.pts[1][e], dims=2)) ./ size(o.pts[1][e], 2) for e in cols]); colroot = hcluster(XF, collect(1:length(cols)), 48)
imgF = [[BC.reflect_point(t, SVector{3,Float64}(XF[:, j])) for j in axes(XF, 2)] for t in trs]
hpad = maximum(sqrt(2 * e.area) for e in op.elements)
function admc(a, b, eta)
    db = hdiam(b) + 2hpad
    min(hdiam(a), db) <= eta * hdist(a, HCluster(b.idx, b.lo .- hpad, b.hi .+ hpad, HCluster[])) || return false
    for iv in imgF
        P = reduce(hcat, iv[b.idx]); bb = HCluster(b.idx, vec(minimum(P, dims=2)) .- hpad, vec(maximum(P, dims=2)) .+ hpad, HCluster[])
        min(hdiam(a), db) <= eta * hdist(a, bb) || return false
    end
    true
end
tC = @elapsed HC = hbuild_c_gpu(o, op, g, todev, root, colroot, cols, (a, b) -> admc(a, b, 1.0), tol)
# interface block check against the engine's dense rows: engine block = +C * bem_flux
Cq = zeros(ComplexF64, length(r), length(Q))
let Hr = HC   # rows r of H(-C) * flux, via unit vectors through hmul would be costly; use dense rows of HC blocks
    full = zeros(ComplexF64, length(r), length(cols))
    for b in HC.dense; rr = findall(in(r), b.rows); isempty(rr) || (full[b.rows[rr], b.cols] .+= b.M[rr, :]); end
    for b in HC.lr; rr = findall(in(r), b.rows); isempty(rr) || (full[b.rows[rr], b.cols] .+= b.U[rr, :] * b.V'); end
    Cq .= -(full * Matrix(flux[colsdof, :]))
end
@printf("   interface rows vs engine dense: rel err %.1e\n", norm(Cq - d.bem_iface_rows) / norm(d.bem_iface_rows))
@printf("   H(A) %.3f of dense (%.2f GB c64), H(C) %d cols %.3f of dense (%.2f GB c64); build A %.0fs C %.0fs\n",
    hstorage(HA) / nB^2, hstorage(HA) * 8 / 2^30, length(cols), hstorage(HC) / (nB * length(cols)), hstorage(HC) * 8 / 2^30, tA, tC)
rest = d.rest
function hmul_coupled(x)
    y = rest * x
    data = flux * x[Q]; isempty(Mr) || (data .+= mflux * x[Mr])
    y[B] .+= hmul(HA, x[B]) .- hmul(HC, data[colsdof])
    return y
end
SOL = FmmCoupledSolve(op, rest, B, Q, Mr, flux, mflux; rtol=1e-9)
D = Diagonal(SOL.row_scale)
Am = LinearMap{ComplexF64}(v -> D * hmul_coupled(v), n, n); Pm = LinearMap{ComplexF64}(v -> SOL.precond \ (D \ v), n, n)
for c in axes(d.rhs, 2)
    b = d.rhs[:, c]; xs = d.dense_solution[:, c]
    t = @elapsed x, st = gmres(Am, D * b; N=Pm, ldiv=false, rtol=1e-8, atol=0.0, memory=300, itmax=300)
    @printf("   excitation %d: %d its (%s) %.1fs; max |err|/max|ref|:", c, st.niter, st.solved, t)
    for (name, rr) in (("BEM p", B), ("q", Q), ("mech", Mr), ("elec", Er))
        isempty(rr) && continue
        @printf(" %s %.1e", name, maximum(abs.(x[rr] - xs[rr])) / maximum(abs.(xs[rr])))
    end
    @printf("; rel L2 all %.1e", norm(x - xs) / norm(xs))
    # scaled residuals of both solutions against the H operator: tells conditioning apart from a model mismatch
    @printf("; scaled residual vs H: H sol %.1e, dense sol %.1e\n", norm(D * (b - hmul_coupled(x))) / norm(D * b), norm(D * (b - hmul_coupled(xs))) / norm(D * b))
end

# ---- a-posteriori error estimate prototype (env HMAT_TRUTH_DIR = Float64 dump of the same frequency) ----
# estimate: residual of the dumped solution against the independent H operator, one correction solve
# M_H d = r (loose tolerance / few iterations), d ~ x_true - x_dumped. Compared with the real error vs Float64.
# Scope: only the BEM operator rows (A and C) are re-evaluated independently (H operator, Float64 near
# entries). `rest` (FEM, interface, transducer rows) and the right-hand side come from the dump being
# checked, so errors in those blocks are NOT seen by this estimate.
# A verified reference must be a full Float64 run of the same physical model: Float64 model data,
# CPU (Float64) BEM assembly, Float64 dense matrix and factor. The model fingerprint compares the BEM
# mesh (topology and tags exactly, coordinates up to Float32 rounding), the non-BEM-operator part of the
# system (FEM, interface, transducer rows: coil resistance etc.), the interface operator (sizes must
# match) and the wavenumber (medium, up to Float32 rounding); the right-hand sides are compared per excitation (explicit zero handling).
function check_truth(truth, d)
    meta(x, key) = hasproperty(x, key) ? string(getproperty(x, key)) : "missing"
    unchecked = get(ENV, "HMAT_TRUTH_UNCHECKED", "0") == "1"
    problems = String[]
    for (key, want) in ((:model_scalar_type, "Float64"), (:bem_backend, "cpu"), (:assembly_eltype, "ComplexF64"),
                        (:solver_scalar_type, "Float64"), (:solution_eltype, "ComplexF64"))
        meta(truth, key) == want || push!(problems, "truth $key = $(meta(truth, key)) ($want required)")
    end
    startswith(meta(truth, :dense_solver), "lu_float64") ||
        push!(problems, "truth dense_solver = $(meta(truth, :dense_solver)) (a Float64 LU required)")
    truth.frequency_hz == d.frequency_hz || push!(problems, "frequency $(truth.frequency_hz) vs $(d.frequency_hz)")
    # wavenumber = what the BEM operator is built with: catches a different sound speed (medium) at the same
    # frequency; tolerance allows only the Float32 rounding of the same input (k = 2*pi*f/c)
    kt, kd = Float64(truth.wavenumber), Float64(d.wavenumber)
    abs(kt - kd) > 8 * eps(Float32) * max(abs(kt), abs(kd)) &&
        push!(problems, "wavenumber $(kt) vs $(kd) (different sound speed/medium?)")
    reldiff(a, b) = (na = max(norm(a), norm(b)); iszero(na) ? 0.0 : norm(a - b) / na)
    # geometry: same BEM topology and tags exactly; coordinates equal up to Float32 rounding of the same input
    mt, md = truth.bem_mesh, d.bem_mesh
    if length(mt.vertices) != length(md.vertices) || mt.faces != md.faces || mt.physical_tags != md.physical_tags
        push!(problems, "BEM mesh topology/tags differ ($(length(mt.vertices)) vs $(length(md.vertices)) vertices)")
    else
        extent = maximum(v -> maximum(abs, v), mt.vertices)
        shift = maximum(i -> maximum(abs, mt.vertices[i] - md.vertices[i]), eachindex(mt.vertices))
        shift > 8 * eps(Float32) * extent &&
            push!(problems, "BEM mesh coordinates differ by up to $(shift) (more than Float32 rounding)")
    end
    if size(truth.rest) != size(d.rest)
        push!(problems, "system size $(size(truth.rest)) vs $(size(d.rest))")
    else
        (m = reldiff(truth.rest, d.rest)) > 1e-4 && push!(problems, "model fingerprint: non-BEM system part differs by $(m) (different transducer/FEM data?)")
        if size(truth.bem_flux) != size(d.bem_flux)
            push!(problems, "interface operator size $(size(truth.bem_flux)) vs $(size(d.bem_flux))")
        elseif (m = reldiff(truth.bem_flux, d.bem_flux)) > 1e-4
            push!(problems, "interface operator differs by $(m)")
        end
    end
    if size(truth.rhs) != size(d.rhs)
        push!(problems, "rhs size $(size(truth.rhs)) vs $(size(d.rhs))")
    else
        for c in axes(d.rhs, 2)
            bt = truth.rhs[:, c]; bd = d.rhs[:, c]
            iszero(bt) != iszero(bd) && (push!(problems, "excitation $c: zero in one dump only"); continue)
            (m = reldiff(bt, bd)) > 1e-4 && push!(problems, "excitation $c: right-hand side differs by $(m) (order or model?)")
        end
    end
    if !isempty(problems)
        unchecked || error("HMAT_TRUTH_DIR is not a usable Float64 reference: " * join(problems, "; ") *
                           " (set HMAT_TRUTH_UNCHECKED=1 to compare anyway)")
        println("   WARNING (HMAT_TRUTH_UNCHECKED=1): ", join(problems, "; "))
    end
    return unchecked && !isempty(problems) ? "UNCHECKED reference" : "verified Float64 dump"
end

if haskey(ENV, "HMAT_TRUTH_DIR")
    truth = hmat_load(ENV["HMAT_TRUTH_DIR"])
    ec < ep && (truth = merge(truth, (rhs=conj.(truth.rhs), dense_solution=conj.(truth.dense_solution),
                                      rest=conj.(truth.rest), bem_flux=conj.(truth.bem_flux))))  # as d above
    label = check_truth(truth, d)
    tr = truth.dense_solution
    nexc = min(size(d.rhs, 2), parse(Int, get(ENV, "HMAT_EST_NEXC", "6")))
    println("\n== a-posteriori estimate vs true error (truth: $label)")
    println("   scope: residual of A, C re-evaluated (H operator); rest and rhs taken from the checked dump")
    for its in (3, 15), c in 1:nexc
        b = d.rhs[:, c]; xs = d.dense_solution[:, c]; xt = tr[:, c]
        r = b - hmul_coupled(xs)
        t = @elapsed dl, st = gmres(Am, D * r; N=Pm, ldiv=false, rtol=1e-3, atol=0.0, memory=300, itmax=its)
        @printf("   its<=%3d exc %2d (%3d its %.1fs):", its, c, st.niter, t)
        for (name, rr) in (("p", B), ("q", Q), ("mech", Mr), ("elec", Er))
            isempty(rr) && continue
            e_true = maximum(abs.(xt[rr] - xs[rr])) / maximum(abs.(xt[rr])); e_est = maximum(abs.(dl[rr])) / maximum(abs.(xs[rr]))
            mis = norm(dl[rr] - (xt[rr] - xs[rr])) / max(norm(xt[rr] - xs[rr]), eps())
            @printf(" %s true %.1e est %.1e (shape miss %.2f)", name, e_true, e_est, mis)
        end
        println()
    end
end

# ---- row-sampling assembly indicator prototype (env HMAT_ROWSAMPLE=1) ----
# Cheap alternative to the full H residual: only the s dumped BEM rows are re-evaluated exactly (Float64),
# Δr_s = (A64 - A_engine)[rows, :] x. Full residual norm extrapolated as sqrt(nB/s) |Δr_s|; the error is
# propagated with the amplification of the solver on a random vector of that norm.
# Compared with the full H residual (all rows) and with the true error vs the Float64 dump.
if get(ENV, "HMAT_ROWSAMPLE", "0") == "1"
    s = length(r)
    Iq = d.bem_iface_rows   # engine interface rows (already conjugated above when needed)
    Ar = d.bem_lhs_rows
    ec < ep && (Ar = conj.(Ar))
    Ex = ec < ep ? E : E      # exact rows in the oracle convention (d is converted to it above)
    rng = MersenneTwister(5)
    println("\n== row-sampling assembly indicator (s = $s of $nB BEM rows)")
    for c in 1:min(size(d.rhs, 2), parse(Int, get(ENV, "HMAT_EST_NEXC", "4")))
        xs = d.dense_solution[:, c]; b = d.rhs[:, c]
        dr_s = (Ex - Ar) * xs[B] .+ (Cq - Iq) * xs[Q]
        rfull = b - hmul_coupled(xs)                              # full residual vs the H operator
        est_norm = sqrt(nB / s) * norm(dr_s)
        z = randn(rng, ComplexF64, n); z[setdiff(1:n, B)] .= 0; z .*= est_norm / norm(z)
        dz, _ = gmres(Am, D * z; N=Pm, ldiv=false, rtol=1e-3, atol=0.0, memory=100, itmax=15)
        df, _ = gmres(Am, D * rfull; N=Pm, ldiv=false, rtol=1e-3, atol=0.0, memory=100, itmax=15)
        @printf("   exc %d: |Δr| sampled-extrapolated %.2e vs full-H %.2e (BEM rows %.2e);", c, est_norm, norm(rfull), norm(rfull[B]))
        if haskey(ENV, "HMAT_TRUTH_DIR")
            e = tr[:, c] - xs
            @printf(" error rel: true %.1e, full-H correction %.1e, random-direction %.1e", norm(e) / norm(tr[:, c]), norm(df) / norm(xs), norm(dz) / norm(xs))
        end
        println()
    end
end
