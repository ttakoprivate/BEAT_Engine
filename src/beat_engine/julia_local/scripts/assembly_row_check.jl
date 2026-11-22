# Prototype for issue #4: can a few Float64 equation rows tell whether the mixed-precision
# (Float32) assembly needs a full Float64 recompute?
#
# Repository fixture femvolume.msh / exterior_conforming.msh with an electrodynamic transducer,
# standard path (transducer condensation, interface flux elimination). Per frequency:
#
#   A_mix, b_mix   the dense condensed system as assembled under precision=float32 (Float32 BEM,
#                  Float64 border), captured from RefinedDenseLU (BLAB_COUPLED_DENSE_SPLIT=0)
#   A_64,  b_64    the same system assembled entirely in Float64 (captured from the Float64 build;
#                  A_64 is P'LU reconstructed from its Float64 LU, error ~eps64 |L||U|)
#   x_prod         productive solution (Float32 LU + refinement on A_mix)
#   x_mix = A_mix \ b_mix,  x_64 = A_64 \ b_64   (dense Float64 LU)
#
# Three differences per output and excitation (issue #4, comment by the reviewer):
#   solver part   x_prod vs x_mix,  assembly part   x_mix vs x_64,  total   x_prod vs x_64.
# Outputs read directly from the dense unknowns: BEM boundary pressure, diaphragm velocity and
# voice-coil current (FEM pressure and interface flux are reconstructed afterwards; not covered).
#
# Row check: on sampled rows i, r_asm = (b_64 - b_mix) - (A_64 - A_mix) x_prod. Small blocks
# (mechanical, electrical) are checked in full; the BEM block by s rows drawn uniformly without
# replacement (inclusion probability π = s / m_BEM). For a linear output y = gᴴ x, with
# A_mixᴴ z = g (adjoint solve with the existing factor), the Horvitz-Thompson estimate of the
# assembly contribution is  Σ_i conj(z_i) r_asm_i / π_i  over the checked rows; the truth is
# gᴴ (x_64 - x_mix). This prototype takes the rows from the full Float64 build; it measures the
# statistics, not the cost of a row-wise Float64 reassembly (not implemented yet).
# A control injects a relative perturbation into a few BEM rows of A_mix (a local assembly fault).
#
#   julia --project=src/beat_engine/julia_local \
#     src/beat_engine/julia_local/scripts/assembly_row_check.jl [frequency_hz ...]
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl")); using .BeatEngineCoupledCondensed
using LinearAlgebra, StaticArrays, Printf, Random, Statistics

const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures")
const ENV_COMMON = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1",
                    "BLAB_COUPLED_DENSE_SPLIT" => "0", "BLAB_COUPLED_DENSE_REFINEMENT" => "1",
                    "BLAB_COUPLED_DENSE_FLOAT64" => "0")
frequencies = isempty(ARGS) ? [2000.0] : parse.(Float64, ARGS)

function setup(::Type{T}) where {T}
    fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), T(0.001))
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("component:test", [radiator], T[1], [1], T[-1], SVector{3,T}(0, 0, 1),
        T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    return (; fem, bem, map, radiator, transducer)
end

# Records the right-hand side and the solution of the dense solve, delegates everything else.
struct Recorder{F}
    factorization::F
    rhs::Base.RefValue{Any}
    solution::Base.RefValue{Any}
end
Base.size(r::Recorder, dims...) = size(r.factorization, dims...)
Base.:\(r::Recorder, b) = (r.rhs[] = ComplexF64.(b); x = r.factorization \ b; r.solution[] = ComplexF64.(x); x)

function capture(s, ::Type{T}, f) where {T}
    system = withenv(ENV_COMMON...) do
        build_condensed_coupled_system(s.fem, s.bem, s.map, T(f), T(343), T(1.21); transducers=[s.transducer])
    end
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
                   (kind=:normal_velocity, radiator_tag=s.radiator, amplitude=Complex{T}(0, 1))]
    recorder = Recorder(system.factorization, Ref{Any}(nothing), Ref{Any}(nothing))
    withenv(ENV_COMMON...) do
        solve_condensed_coupled_excitations(merge(system, (factorization=recorder,)), excitations)
    end
    F = system.factorization
    A = F isa BeatEngineCoupledCondensed.RefinedDenseLU ? copy(F.matrix) :
        F isa LinearAlgebra.LU ? Matrix(F.P' * (F.L * F.U)) : error("unexpected factorization $(typeof(F))")
    ranges = (bem=system.bem_range, mechanical=system.mechanical_range, electrical=system.electrical_range)
    result = (; A, b=recorder.rhs[], x=recorder.solution[], ranges, solver=typeof(F))
    release_condensed_coupled_system!(system)
    return result
end

pointwise(reference, candidate) = maximum(abs.(candidate .- reference) ./ max.(abs.(reference), floatmin()))

function report_parts(mix, full, x_mix, x_64)
    println("  max pointwise complex relative error (worst entry of each output, per excitation):")
    @printf("  %-12s %-4s %12s %12s %12s\n", "output", "exc", "solver", "assembly", "total")
    for (name, range) in pairs(mix.ranges), k in axes(mix.x, 2)
        isempty(range) && continue
        solver = pointwise(x_mix[range, k], mix.x[range, k])
        assembly = pointwise(x_64[range, k], x_mix[range, k])
        total = pointwise(x_64[range, k], mix.x[range, k])
        @printf("  %-12s %-4d %12.3e %12.3e %12.3e\n", name, k, solver, assembly, total)
    end
end

# Horvitz-Thompson estimate of gᴴ (x_64 - x_mix) for every output g (columns of z) from s BEM rows
function row_estimate(r_asm, z, ranges, s, rng)
    bem = collect(ranges.bem)
    small = vcat(collect(ranges.mechanical), collect(ranges.electrical))
    sampled = bem[randperm(rng, length(bem))[1:s]]
    weight = length(bem) / s
    estimate = transpose(sum(conj.(z[small, :]) .* r_asm[small]; dims=1))
    estimate .+= weight .* transpose(sum(conj.(z[sampled, :]) .* r_asm[sampled]; dims=1))
    return vec(estimate)
end

# Importance sampling (Poisson design): BEM row i is checked with probability
# π_i = min(1, s w_i / Σ w), w_i = |z_i| (|A_mix| |x| + |b_mix|)_i, the expected size of
# |conj(z_i) r_asm_i| if every entry carries a relative assembly error of the same order. Needs only
# the adjoint z and quantities of the mixed system; r_asm is then computed on the checked rows.
function importance_estimate(r_asm, z, weights, ranges, s, rng)
    bem = collect(ranges.bem)
    small = vcat(collect(ranges.mechanical), collect(ranges.electrical))
    w = weights[bem]
    pi = min.(1.0, s .* w ./ sum(w))
    chosen = rand(rng, length(bem)) .< pi
    return sum(conj.(z[small]) .* r_asm[small]) + sum(conj.(z[bem[chosen]]) .* r_asm[bem[chosen]] ./ pi[chosen]), count(chosen)
end

function row_check(mix, full, x_mix, x_64; label, sizes=(16, 64, 256), seeds=1:20)
    n = size(mix.A, 1)
    F = lu(mix.A)
    bem = collect(mix.ranges.bem)
    # outputs: current, velocity, the BEM node with the largest assembly error, two fixed BEM nodes
    worst = bem[argmax(abs.(x_64[bem, 1] .- x_mix[bem, 1]) ./ abs.(x_64[bem, 1]))]
    outputs = vcat(collect(mix.ranges.electrical), collect(mix.ranges.mechanical), [worst, bem[1], bem[end ÷ 2]])
    names = vcat(["current"], ["velocity"], ["p_bem[worst]", "p_bem[1]", "p_bem[mid]"])
    G = zeros(ComplexF64, n, length(outputs))
    for (k, i) in enumerate(outputs); G[i, k] = 1; end
    Z = F' \ G                                                   # adjoint solves A_mixᴴ z = g
    head = bem[1:min(64, length(bem))]
    small = vcat(collect(mix.ranges.mechanical), collect(mix.ranges.electrical))
    for e in axes(mix.x, 2)
        println("  row check ($label), excitation $e: estimate / truth of the assembly contribution gᴴ(x_64 - x_mix)")
        r_asm = (full.b[:, e] .- mix.b[:, e]) .- (full.A .- mix.A) * mix.x[:, e]
        truth = [transpose(G[:, k]) * (x_64[:, e] .- x_mix[:, e]) for k in axes(G, 2)]
        linear = vec(sum(conj.(Z) .* r_asm; dims=1))             # all rows: first-order prediction
        # deterministic baseline: small blocks plus the first 64 BEM rows, extrapolated
        head64 = vec(sum(conj.(Z[small, :]) .* r_asm[small]; dims=1)) .+
                 (length(bem) / length(head)) .* vec(sum(conj.(Z[head, :]) .* r_asm[head]; dims=1))
        @printf("  %-14s %12s %12s", "output", "|truth|", "all rows")
        for s in sizes; @printf(" %22s", "s=$s median [min,max]"); end
        @printf(" %18s %28s\n", "first 64 BEM rows", "importance s=64 med [min,max]")
        for k in axes(G, 2)
            @printf("  %-14s %12.3e %12.3f", names[k], abs(truth[k]), abs(linear[k] / truth[k]))
            for s in sizes
                ratios = [abs(row_estimate(r_asm, Z, mix.ranges, s, MersenneTwister(seed))[k] / truth[k]) for seed in seeds]
                @printf("   %6.2f [%5.2f,%6.2f]", median(ratios), minimum(ratios), maximum(ratios))
            end
            @printf(" %18.2f", abs(head64[k] / truth[k]))
            scale = abs.(mix.A) * abs.(mix.x[:, e]) .+ abs.(mix.b[:, e])
            weights = abs.(Z[:, k]) .* scale
            ratios = [abs(importance_estimate(r_asm, Z[:, k], weights, mix.ranges, 64, MersenneTwister(seed))[1] / truth[k])
                      for seed in seeds]
            @printf("      %6.2f [%5.2f,%6.2f]\n", median(ratios), minimum(ratios), maximum(ratios))
        end
    end
    println("  BEM rows: $(length(bem)) of n = $n; seeds per sample size: $(length(seeds)); ratio 1 = exact.")
end

s32 = setup(Float32); s64 = setup(Float64)
for f in frequencies
    println("\n=== $(f) Hz ===")
    mix = capture(s32, Float32, f)
    full = capture(s64, Float64, f)
    @printf("  mixed solver %s, n = %d, ‖A_64 - A_mix‖_F / ‖A_64‖_F = %.3e, ‖b_64 - b_mix‖ / ‖b_64‖ = %.3e\n",
            mix.solver, size(mix.A, 1), norm(full.A - mix.A) / norm(full.A), norm(full.b - mix.b) / norm(full.b))
    x_mix = lu(mix.A) \ mix.b
    x_64 = lu(full.A) \ full.b
    @printf("  Float64 build: recorded x vs P'LU-reconstructed solve, max rel %.3e (reconstruction check)\n",
            pointwise(full.x, x_64))
    report_parts(mix, full, x_mix, x_64)
    row_check(mix, full, x_mix, x_64; label="as assembled")
    # control: a local assembly fault, random relative error 1e-4 per entry on 3 BEM rows of A_mix
    # (not a pure row scaling, which leaves x unchanged where b vanishes)
    rng = MersenneTwister(99)
    faulty_rows = collect(mix.ranges.bem)[randperm(rng, length(mix.ranges.bem))[1:3]]
    A_fault = copy(mix.A)
    A_fault[faulty_rows, :] .*= 1 .+ 1e-4 .* randn(rng, ComplexF64, 3, size(A_fault, 2))
    x_fault = lu(A_fault) \ mix.b
    fault = (; mix..., A=A_fault, x=x_fault)
    row_check(fault, full, x_fault, x_64; label="control: 3 BEM rows, random 1e-4 relative entry errors")
    @printf("  faulty rows: %s\n", string(faulty_rows))
end
