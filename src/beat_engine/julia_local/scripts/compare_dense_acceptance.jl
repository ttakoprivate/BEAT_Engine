# Accuracy of the refined dense coupled solve (RefinedDenseLU / SplitRefinedLU) with and without
# the componentwise acceptance test (BLAB_COUPLED_DENSE_COMPONENTWISE), against the ComplexF64 LU
# of the same mixed-precision assembled system (solver error only) and against the fully Float64
# build (context). Repository fixture femvolume.msh / exterior_conforming.msh with a transducer.
#
#   julia --project=src/beat_engine/julia_local \
#     src/beat_engine/julia_local/scripts/compare_dense_acceptance.jl [frequency_hz ...]
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl")); using .BeatEngineCoupledCondensed
using LinearAlgebra, StaticArrays, Printf

const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures")
const FIELDS = (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
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

function solve(s, ::Type{T}, f, env...) where {T}
    system = withenv(env...) do
        build_condensed_coupled_system(s.fem, s.bem, s.map, T(f), T(343), T(1.21); transducers=[s.transducer])
    end
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
                   (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(0, 2))]
    try
        t = @elapsed solutions = withenv(env...) do
            solve_condensed_coupled_excitations(system, excitations)
        end
        return solutions, BeatEngineCoupledCondensed.dense_solver_diagnostics(system), t
    finally
        release_condensed_coupled_system!(system)
    end
end

# max pointwise complex relative error where |reference| > 1e-6 max|reference|, and max abs error / max|reference|
function pointwise(reference, candidate)
    r = ComplexF64.(collect(reference)); c = ComplexF64.(collect(candidate))
    peak = maximum(abs, r; init=0.0)
    mask = abs.(r) .> 1e-6 * peak
    rel = any(mask) ? maximum(abs.(c[mask] .- r[mask]) ./ abs.(r[mask])) : NaN
    return rel, peak > 0 ? maximum(abs, c .- r) / peak : 0.0
end

common = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1")
s32 = setup(Float32); s64 = setup(Float64)
for f in frequencies
    reference, _, _ = solve(s32, Float32, f, common..., "BLAB_COUPLED_DENSE_REFINEMENT" => "0", "BLAB_COUPLED_DENSE_FLOAT64" => "1")
    full64, _, _ = solve(s64, Float64, f, common...)
    for split in ("auto", "0"), componentwise in ("0", "1")
        candidate, diagnostics, t = solve(s32, Float32, f, common..., "BLAB_COUPLED_DENSE_REFINEMENT" => "1",
                                          "BLAB_COUPLED_DENSE_FLOAT64" => "0", "BLAB_COUPLED_DENSE_SPLIT" => split,
                                          "BLAB_COUPLED_DENSE_COMPONENTWISE" => componentwise)
        @printf("\n%.0f Hz split=%s componentwise=%s solver=%s iterations=%d fallback=%s normwise=%.3g omega=%s tol=%s solve=%.2fs\n",
                f, split, componentwise, diagnostics["dense_solver"], diagnostics["dense_refinement_iterations"],
                something(diagnostics["dense_refinement_fallback_reason"], "none"), diagnostics["dense_refinement_backward_error"],
                string(diagnostics["dense_refinement_componentwise_backward_error"]), string(diagnostics["dense_refinement_componentwise_tolerance"]), t)
        for field in FIELDS, k in eachindex(candidate)
            rel, ab = pointwise(getproperty(reference[k], field), getproperty(candidate[k], field))
            rel64, ab64 = pointwise(getproperty(full64[k], field), getproperty(candidate[k], field))
            @printf("  exc%d %-20s vs F64-LU(same A): rel %.3e abs/peak %.3e | vs full F64 build: rel %.3e abs/peak %.3e\n",
                    k, field, rel, ab, rel64, ab64)
        end
    end
end
