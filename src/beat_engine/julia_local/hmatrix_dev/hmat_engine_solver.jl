# ⚠️ WORK IN PROGRESS (roadmap step 6, part 2): H-matrix BEM solver for the engine hook
# `build_condensed_coupled_system(...; bem_solver=hmatrix_bem_solver())`.
#
# The hook (src/BeatEngineCoupledCondensed.jl) never assembles the dense Burton-Miller blocks; it
# hands this solver the sparse system `rest` without the BEM operator entries. Here the BEM rows are
# applied as H(A) x_B + H(C) (bem_flux x_q + bem_motion_flux x_mech) and the coupled system is solved
# with GMRES and the bordered near-field preconditioner of `BeatEngineFmm.FmmCoupledSolve`
# (exact near blocks). Same code as the offline prototype `tests/hmat_coupled_proto.jl`, wired into
# the hook contract. Not reachable from the engine's normal solve paths and from Boundary Lab.
#
# Load order (like the prototype): BeatEngineCore, BeatEngineCoupled(+Condensed), BeatEngineFmm
# (needs FMM3D, Krylov, LinearMaps; KernelAbstractions and Adapt for hmat_ka.jl), then this file.
using LinearAlgebra, SparseArrays, StaticArrays, Printf, Krylov, LinearMaps
isdefined(@__MODULE__, :BC) || (const BC = BeatEngineCore)  # alias used by hmat_fast.jl
isdefined(@__MODULE__, :HMat) || include(joinpath(@__DIR__, "hmat_core.jl"))
isdefined(@__MODULE__, :FastOracle) || include(joinpath(@__DIR__, "hmat_fast.jl"))
isdefined(@__MODULE__, :KAGeom) || include(joinpath(@__DIR__, "hmat_ka.jl"))
isdefined(@__MODULE__, :hbuild_c_gpu) || include(joinpath(@__DIR__, "hmat_c.jl"))

# Admissibility min(diam a, diam b) <= eta dist(a, b) against the cluster and all its symmetry images.
function _hmat_admissible(images, pad::Float64)
    return (a, b, eta) -> begin
        db = hdiam(b) + 2pad
        min(hdiam(a), db) <= eta * hdist(a, HCluster(b.idx, b.lo .- pad, b.hi .+ pad, HCluster[])) || return false
        for iv in images
            P = reduce(hcat, iv[b.idx])
            bb = HCluster(b.idx, vec(minimum(P, dims=2)) .- pad, vec(maximum(P, dims=2)) .+ pad, HCluster[])
            min(hdiam(a), db) <= eta * hdist(a, bb) || return false
        end
        true
    end
end

"""Coupled solve object for the hook: `size` and `\\` like `SplitGMRES` / `FmmCoupledSolve`."""
struct HMatCoupledSolve
    base::FmmCoupledSolve          # rest, ranges, flux maps, bordered near preconditioner, row scale
    HA::HMat{ComplexF64}           # H(A), P1 x P1
    HC::HMat{ComplexF64}           # H(-C) on the DP0 columns `colsdof` (BeatEngineFmm sign convention)
    colsdof::Vector{Int}
    stats::Vector{NamedTuple}
    build::NamedTuple              # build diagnostics (storage ratios, block counts, seconds)
end
Base.size(S::HMatCoupledSolve, d::Integer) = size(S.base, d)
Base.size(S::HMatCoupledSolve) = size(S.base)

function _hmat_coupled_mul(S::HMatCoupledSolve, x::AbstractVector)
    b = S.base
    y = b.rest * x
    data = isempty(b.Q) ? zeros(ComplexF64, size(b.flux, 1)) : b.flux * x[b.Q]
    isempty(b.M) || (data .+= b.motion_flux * x[b.M])
    y[b.B] .+= hmul(S.HA, x[b.B]) .- hmul(S.HC, data[S.colsdof])
    return y
end

function Base.:\(S::HMatCoupledSolve, rhs::AbstractVecOrMat)
    n = size(S, 1); b = S.base
    D = Diagonal(b.row_scale)
    A = LinearMap{ComplexF64}(v -> D * _hmat_coupled_mul(S, v), n, n)
    P = LinearMap{ComplexF64}(v -> b.precond \ (D \ v), n, n)
    R = rhs isa AbstractVector ? reshape(rhs, :, 1) : rhs
    X = zeros(ComplexF64, n, size(R, 2))
    failed = Int[]; failed_residuals = Float64[]
    for c in axes(R, 2)
        r = D * ComplexF64.(R[:, c])
        if iszero(r)
            push!(S.stats, (iterations=0, solved=true, residual=0.0, seconds=0.0)); continue
        end
        t = @elapsed x, st = gmres(A, r; N=P, ldiv=false, rtol=b.rtol, atol=0.0, memory=min(b.maxiter, 300),
            itmax=b.maxiter, restart=b.maxiter > 300)
        X[:, c] = x
        push!(S.stats, (iterations=st.niter, solved=st.solved, residual=norm(r - A * x) / norm(r), seconds=t))
        st.solved || (push!(failed, c); push!(failed_residuals, S.stats[end].residual))
    end
    isempty(failed) || throw(FmmNotConvergedError(failed, failed_residuals, X))
    return rhs isa AbstractVector ? vec(X) : X
end

"""
    hmatrix_bem_solver(; tol=1e-6, eta=1.0, leaf=48, rtol=1e-9, maxiter=300, entry_type=Float64,
                       todev=identity, solves=nothing)

Returns a `bem_solver` for `build_condensed_coupled_system`. `tol`: ACA/recompression tolerance;
`eta`: admissibility; `leaf`: cluster leaf size; `rtol`: GMRES relative residual of the row-scaled
system. `entry_type`/`todev` select the far-field entry kernel (Float64 CPU here; Float32 + device
array constructor for Metal/CUDA). If `solves` is a vector, every `HMatCoupledSolve` is pushed to it
(iterations, residuals and build diagnostics for tests and reports). The returned hook also has
`exterior_solve` (A \\ R on H(A), used by `retain_interface_radiation`); if `exterior_stats` is a vector,
each of its GMRES columns pushes `(iterations, solved, residual, seconds)` there.
"""
function hmatrix_bem_solver(; tol::Float64=1e-6, eta::Float64=1.0, leaf::Int=48, rtol::Float64=1e-9,
        maxiter::Int=300, entry_type::Type=Float64, todev=identity, solves=nothing, exterior_stats=nothing)
    return context -> begin
        T = context.scalar_type
        T === Float64 || error("The H-matrix BEM solver is validated with Float64 systems only (got $T).")
        bem = context.bem_mesh; prepared = context.prepared; k = context.wavenumber
        symmetry = context.symmetry_mode
        p1 = prepared.p1; dp0 = prepared.dp0; rule = prepared.rule
        t_op = @elapsed op = fmm_burton_miller_operator(bem, p1, dp0, k, rule;
            identity_p1_p1=prepared.identity_p1_p1, identity_p1_dp0=prepared.identity_p1_dp0,
            singular_order=context.singular_order, symmetry_mode=symmetry, eps=1e-10)
        o = FastOracle(bem, p1, dp0, k, rule, op, symmetry)
        g = ka_geom(o, entry_type, todev)
        nB = length(bem.vertices)
        X = reduce(hcat, bem.vertices); root = hcluster(X, collect(1:nB), leaf)
        transforms = BeatEngineCore.symmetry_image_transforms(BeatEngineCore.normalized_symmetry_mode(symmetry))
        imgV = [[BeatEngineCore.reflect_point(t, v) for v in bem.vertices] for t in transforms]
        admA = _hmat_admissible(imgV, 0.0)
        t_A = @elapsed (HA, _, _) = hbuild_gpu(o, g, todev, root, (a, b) -> admA(a, b, eta), tol, nB)
        dofof = [el.dp0_dof for el in op.elements]
        # C * Q for prescribed BEM normal data: exact direct sum (hook sign: C = -fmm_rhs).
        flux_operator_mul = Q -> begin
            out = zeros(ComplexF64, nB, size(Q, 2))
            for c in axes(Q, 2)
                iszero(Q[:, c]) && continue
                out[:, c] .= .-rhs_direct(op, ComplexF64.(Q[:, c:c]))
            end
            out
        end
        factorize = (rest, layout) -> begin
            B = collect(layout.bem_range); Qr = collect(layout.flux_range)
            Mr = collect(layout.mechanical_range)
            flux = sparse(ComplexF64.(layout.bem_flux))
            motion = isempty(Mr) ? spzeros(ComplexF64, size(flux, 1), 0) : sparse(ComplexF64.(layout.bem_motion_flux))
            used = falses(dp0.global_dof_count)
            used[unique(rowvals(flux))] .= true
            isempty(Mr) || (used[unique(rowvals(motion))] .= true)
            cols = [fc for fc in eachindex(op.elements) if used[dofof[fc]]]
            isempty(cols) && error("The H-matrix BEM solver needs at least one face carrying interface or motion flux.")
            colsdof = dofof[cols]
            XF = reduce(hcat, [vec(sum(o.pts[1][e], dims=2)) ./ size(o.pts[1][e], 2) for e in cols])
            colroot = hcluster(XF, collect(1:length(cols)), leaf)
            imgF = [[BeatEngineCore.reflect_point(t, SVector{3,Float64}(XF[:, j])) for j in axes(XF, 2)] for t in transforms]
            hpad = maximum(sqrt(2 * e.area) for e in op.elements)
            admC = _hmat_admissible(imgF, hpad)
            t_C = @elapsed HC = hbuild_c_gpu(o, op, g, todev, root, colroot, cols, (a, b) -> admC(a, b, eta), tol)
            t_P = @elapsed base = FmmCoupledSolve(op, rest, B, Qr, Mr, flux, motion; rtol=rtol, maxiter=maxiter)
            build = (operator_s=t_op, build_A_s=t_A, build_C_s=t_C, preconditioner_s=t_P,
                     storage_A=hstorage(HA) / nB^2, storage_C=hstorage(HC) / (nB * length(cols)),
                     flux_columns=length(cols), tol=tol, eta=eta, leaf=leaf, rtol=rtol)
            S = HMatCoupledSolve(base, HA, HC, colsdof, NamedTuple[], build)
            isnothing(solves) || push!(solves, S)
            S
        end
        # A \ R for the interface radiation replay (`retain_interface_radiation`): GMRES on H(A) alone,
        # right-preconditioned with the near-field LU of A and row-scaled like the coupled solve.
        exterior_solve = R -> begin
            An, _ = fmm_near_blocks(op)
            F = lu(An)
            D = Diagonal(1 ./ max.(vec(sqrt.(sum(abs2, An; dims=2))), floatmin(Float64)))
            Aop = LinearMap{ComplexF64}(v -> D * hmul(HA, v), nB, nB)
            P = LinearMap{ComplexF64}(v -> F \ (D \ v), nB, nB)
            X = zeros(ComplexF64, nB, size(R, 2))
            failed = Int[]; failed_residuals = Float64[]
            for c in axes(R, 2)
                r = D * ComplexF64.(R[:, c])
                if iszero(r)
                    isnothing(exterior_stats) || push!(exterior_stats, (iterations=0, solved=true, residual=0.0, seconds=0.0))
                    continue
                end
                t = @elapsed x, st = gmres(Aop, r; N=P, ldiv=false, rtol=rtol, atol=0.0, memory=min(maxiter, 300),
                    itmax=maxiter, restart=maxiter > 300)
                X[:, c] = x
                residual = norm(r - Aop * x) / norm(r)
                isnothing(exterior_stats) ||
                    push!(exterior_stats, (iterations=st.niter, solved=st.solved, residual=residual, seconds=t))
                st.solved || (push!(failed, c); push!(failed_residuals, residual))
            end
            isempty(failed) || throw(FmmNotConvergedError(failed, failed_residuals, X))
            X
        end
        (; flux_operator_mul, factorize, exterior_solve)
    end
end

"""
    hmatrix_options_from_env(env=ENV)

H-matrix settings for the driver switch `BLAB_COUPLED_BEM_SOLVER=hmatrix`: `BLAB_HMATRIX_TOL` (1e-6),
`BLAB_HMATRIX_ETA` (1.0), `BLAB_HMATRIX_LEAF` (48), `BLAB_HMATRIX_RTOL` (1e-9), `BLAB_HMATRIX_MAXITER`
(300). The defaults are the values of the hook tests; they are not a qualified accuracy setting.
"""
function hmatrix_options_from_env(env=ENV)
    read(name, T, default) = begin
        raw = strip(String(get(env, name, "")))
        isempty(raw) && return default
        value = tryparse(T, raw)
        (isnothing(value) || !(value > 0) || (T === Float64 && !isfinite(value))) &&
            throw(ArgumentError("$name must be a positive $(T === Int ? "integer" : "finite number"), got \"$raw\"."))
        value
    end
    return (tol=read("BLAB_HMATRIX_TOL", Float64, 1e-6), eta=read("BLAB_HMATRIX_ETA", Float64, 1.0),
            leaf=read("BLAB_HMATRIX_LEAF", Int, 48), rtol=read("BLAB_HMATRIX_RTOL", Float64, 1e-9),
            maxiter=read("BLAB_HMATRIX_MAXITER", Int, 300))
end

"""
    hmatrix_external_bem_solver(; options...)

Factory result for `register_external_bem_solver!`: `(; bem_solver, diagnostics)`. `diagnostics()`
reports the frequency solved since its previous call (build seconds and storage ratios of H(A) and
H(C), GMRES iterations, residuals and seconds per right-hand side, and the same for the exterior
replay solve) and then forgets it, so every frequency carries only its own values.
"""
function hmatrix_external_bem_solver(; options...)
    solves = HMatCoupledSolve[]
    exterior = NamedTuple[]
    hook = hmatrix_bem_solver(; options..., solves=solves, exterior_stats=exterior)
    diagnostics = () -> begin
        isempty(solves) && isempty(exterior) && return nothing
        d = Dict{String,Any}("solver" => "hmatrix")
        if !isempty(solves)
            S = solves[end]
            d["build"] = Dict{String,Any}(String(k) => v for (k, v) in pairs(S.build))
            d["gmres_iterations"] = [s.iterations for s in S.stats]
            d["gmres_relative_residuals"] = [s.residual for s in S.stats]
            d["gmres_solved"] = [s.solved for s in S.stats]
            d["gmres_s"] = [s.seconds for s in S.stats]
        end
        d["exterior_gmres_iterations"] = [s.iterations for s in exterior]
        d["exterior_gmres_relative_residuals"] = [s.residual for s in exterior]
        d["exterior_gmres_s"] = [s.seconds for s in exterior]
        empty!(solves); empty!(exterior)
        d
    end
    return (; bem_solver=hook, diagnostics)
end

# Loading this file makes `BLAB_COUPLED_BEM_SOLVER=hmatrix` available to the driver; settings are read
# from the environment once per request (`hmatrix_options_from_env`).
BeatEngineCoupledCondensed.register_external_bem_solver!("hmatrix",
    () -> hmatrix_external_bem_solver(; hmatrix_options_from_env()...))
