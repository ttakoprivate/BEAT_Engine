# Step 2 of docs/ROADMAP.md on the repository 2-kHz fixture (femvolume.msh / exterior_conforming.msh
# with a transducer): estimated complex relative solver error per output and excitation
# (dense_output_estimated_relative_error_*) against the true error of the Float32-refined solve vs
# (a) the ComplexF64 LU of the same mixed-precision assembled system (solver error, what the estimate
#     covers) and (b) the fully Float64 build (includes the Float32 assembly error of step 4, which
#     the estimate does not cover; reported for context only).
# A "false release" is an entry with estimate < true error (a).
#
#   julia --project=src/beat_engine/julia_local \
#     src/beat_engine/julia_local/scripts/validate_output_error_estimate.jl [frequency_hz ...]
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl")); using .BeatEngineCoupledCondensed
using LinearAlgebra, StaticArrays, Printf

const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures")
const OUTPUTS = (:diaphragm_velocity, :voice_coil_current)
frequencies = isempty(ARGS) ? [2000.0] : parse.(Float64, ARGS)

function setup(::Type{T}) where {T}
    fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), T(0.001))
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("component:test", [radiator], T[1], [1], T[-1], SVector{3,T}(0, 0, 1),
        T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    return (; fem, bem, map, transducer)
end

function solve(s, ::Type{T}, f, env...) where {T}
    system = withenv(env...) do
        build_condensed_coupled_system(s.fem, s.bem, s.map, T(f), T(343), T(1.21); transducers=[s.transducer])
    end
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
                   (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(0, 2))]
    try
        solutions = withenv(env...) do
            solve_condensed_coupled_excitations(system, excitations)
        end
        return solutions, BeatEngineCoupledCondensed.dense_solver_diagnostics(system)
    finally
        release_condensed_coupled_system!(system)
    end
end

relative(c, r) = iszero(r) ? (iszero(c) ? 0.0 : Inf) : abs(ComplexF64(c) - ComplexF64(r)) / abs(ComplexF64(r))

common = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1")
s32 = setup(Float32); s64 = setup(Float64)
false_releases = 0; entries = 0; worst_margin = Inf
for f in frequencies
    same_a, _ = solve(s32, Float32, f, common..., "BLAB_COUPLED_DENSE_REFINEMENT" => "0", "BLAB_COUPLED_DENSE_FLOAT64" => "1")
    full64, _ = solve(s64, Float64, f, common...)
    for split in ("auto", "0")
        candidate, diagnostics = solve(s32, Float32, f, common..., "BLAB_COUPLED_DENSE_REFINEMENT" => "1",
                                       "BLAB_COUPLED_DENSE_FLOAT64" => "0", "BLAB_COUPLED_DENSE_SPLIT" => split)
        @printf("\n%.0f Hz split=%s solver=%s normwise estimate=%s hint=%s\n", f, split, diagnostics["dense_solver"],
                string(get(diagnostics, "dense_estimated_relative_error", nothing)),
                string(get(diagnostics, "dense_output_recompute_float64", nothing)))
        for field in OUTPUTS
            estimates = diagnostics["dense_output_estimated_relative_error_$(field)"]
            for k in eachindex(candidate), i in eachindex(getproperty(candidate[k], field))
                est = something(estimates[k][i], NaN)
                c = getproperty(candidate[k], field)[i]
                solver = relative(c, getproperty(same_a[k], field)[i])
                total = relative(c, getproperty(full64[k], field)[i])
                global entries += 1
                isnan(est) || est >= solver || (global false_releases += 1)
                solver > 0 && (global worst_margin = min(worst_margin, est / solver))
                @printf("  exc%d %-19s [%d] estimate %.3e | true vs F64-LU(same A) %.3e | vs full F64 build %.3e\n",
                        k, field, i, est, solver, total)
            end
        end
    end
end
@printf("\nentries %d, false releases (estimate < true solver error) %d, smallest estimate/true %.3g\n",
        entries, false_releases, worst_margin)
