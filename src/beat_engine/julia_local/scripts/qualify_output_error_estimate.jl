# Step 2 of docs/ROADMAP.md, qualification protocol on the repository 2-kHz fixture (femvolume.msh /
# exterior_conforming.msh with a transducer, two voltage excitations), display only. For a frequency sweep
# and every output (diaphragm velocity, voice-coil current, area-averaged interface normal velocity and
# FIELD_POINTS points on a 1-m arc), the estimated complex relative error of the Float32-refined solve is
# compared with its TRUE complex relative error against the fully Float64 build (assembly and solve in
# Float64). That true error includes the Float32 assembly error of step 4, which the step-2 estimate does
# not cover by construction; the protocol therefore measures which fixed safety factor the estimate would
# need, as the roadmap asks:
#
#   * calibration set = every other sweep frequency (1st, 3rd, ...), holdout = the rest (interleaved, so
#     both span the band); the factor k = max(true / estimate) over the calibration entries (at least 1);
#   * "false release" = an entry whose (factored) estimate is below its true error; reported for k = 1 and
#     for the calibrated k on the holdout;
#   * entries at resonances (sweep frequencies where |diaphragm velocity| of the Float64 build is a local
#     maximum) and at cancellations (field points more than 20 dB below the largest |p| on the arc for that
#     frequency and excitation) are reported separately;
#   * exact-zero references are reported as absolute errors with unit, never as relative ones.
#
# Nothing here changes a solution or a baseline. Not a qualification of h343 (that needs the Mac) and not
# a performance measurement.
#
#   julia --project=src/beat_engine/julia_local \
#     src/beat_engine/julia_local/scripts/qualify_output_error_estimate.jl [frequency_hz ...]
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl")); using .BeatEngineCoupledCondensed
include(joinpath(@__DIR__, "..", "src", "BeatEngineInterfaceVelocity.jl")); using .BeatEngineInterfaceVelocity
using LinearAlgebra, StaticArrays, Printf

const OE = BeatEngineCoupledCondensed
const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures")
const DEFAULT_SWEEP = [300.0, 400.0, 500.0, 700.0, 1000.0, 1400.0, 2000.0, 2800.0, 4000.0, 5000.0, 6000.0, 7000.0]
const FIELD_POINTS = [SVector(sin(t), 0.0, cos(t)) for t in range(0, pi; length=13)]
const CANCELLATION_DB = 20.0
const SPLITS = ("auto", "0")
const COMMON = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1")
const DENSITY = 1.21
frequencies = isempty(ARGS) ? DEFAULT_SWEEP : sort(parse.(Float64, ARGS))

function setup(::Type{T}) where {T}
    fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), T(0.001))
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("component:test", [radiator], T[1], [1], T[-1], SVector{3,T}(0, 0, 1),
        T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    n = length(map.fem_vertex_indices)
    weights = interface_average_weights(fem, [map], [1:n], T).weights
    return (; fem, bem, map, transducer, ranges=[1:n], weights)
end

EXCITATIONS(T) = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
               (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(0, 2))]

# Solve one configuration; returns solutions and (for the candidate) all estimates while the system lives.
function solve(s, ::Type{T}, f, env...; estimates=false) where {T}
    system = withenv(env...) do
        build_condensed_coupled_system(s.fem, s.bem, s.map, T(f), T(343), T(DENSITY); transducers=[s.transducer])
    end
    try
        solutions = withenv(env...) do
            solve_condensed_coupled_excitations(system, EXCITATIONS(T))
        end
        estimates || return (; solutions, system.regular_quadrature_order)
        fields = [evaluate_galerkin_field_cpu(FIELD_POINTS, s.bem, x.bem_pressure, x.bem_neumann,
                                              system.wavenumber, system.field_cache) for x in solutions]
        field = OE.field_output_error_estimates(system, FIELD_POINTS, fields, system.field_cache)
        diagnostics = OE.dense_solver_diagnostics(system)
        scale = neumann_scale(system.density, system.omega)
        fluxes = [x.interface_flux for x in solutions]
        interface = OE.interface_velocity_error_estimates(system, fluxes, s.ranges, s.weights, scale,
                                                          neumann_scale(DENSITY, 2pi * f))
        velocity = [interface_average_normal_velocity(q, s.ranges, s.weights, scale) for q in fluxes]
        return (; solutions, diagnostics, field, interface, velocity, wavenumber=system.wavenumber,
                rule=system.regular_quadrature_order)
    finally
        release_condensed_coupled_system!(system)
    end
end

# one record per entry: output, frequency, excitation, index, estimate, true error (relative, or absolute
# with unit when the reference is exactly zero), magnitude of the reference
records = NamedTuple[]
function record!(output, f, split, exc, i, estimate, candidate, reference)
    c = ComplexF64(candidate); r = ComplexF64(reference)
    zero_ref = iszero(r)
    err = zero_ref ? abs(c) : abs(c - r) / abs(r)
    push!(records, (; output, f, split, exc, i, estimate=Float64(estimate), err, zero_ref, magnitude=abs(r)))
end

s32 = setup(Float32); s64 = setup(Float64)
for f in frequencies
    full64 = solve(s64, Float64, f, COMMON...).solutions
    k64 = 2pi * f / 343
    scale64 = neumann_scale(DENSITY, 2pi * f)
    for split in SPLITS
        c = solve(s32, Float32, f, COMMON..., "BLAB_COUPLED_DENSE_REFINEMENT" => "1",
                  "BLAB_COUPLED_DENSE_FLOAT64" => "0", "BLAB_COUPLED_DENSE_SPLIT" => split; estimates=true)
        cache64 = build_field_evaluation_cache(s64.bem, triangle_rule(Float64, c.rule))
        @printf("%.0f Hz split=%s solver=%s\n", f, split, string(c.diagnostics["dense_solver"]))
        for field in (:diaphragm_velocity, :voice_coil_current)
            est = c.diagnostics["dense_output_estimated_relative_error_$(field)"]
            for k in eachindex(c.solutions), i in eachindex(getproperty(c.solutions[k], field))
                record!(field, f, split, k, i, something(est[k][i], NaN), getproperty(c.solutions[k], field)[i],
                        getproperty(full64[k], field)[i])
            end
        end
        for k in eachindex(c.solutions)
            v64 = interface_average_normal_velocity(ComplexF64.(full64[k].interface_flux), s64.ranges, s64.weights, scale64)
            for i in eachindex(v64)
                record!(:interface_velocity, f, split, k, i, c.interface[i, k], c.velocity[k][i], v64[i])
            end
            # both fields in Float64 from the stored boundary data of each solve
            pc = evaluate_galerkin_field_cpu(FIELD_POINTS, s64.bem, ComplexF64.(c.solutions[k].bem_pressure),
                                             ComplexF64.(c.solutions[k].bem_neumann), k64, cache64)
            pr = evaluate_galerkin_field_cpu(FIELD_POINTS, s64.bem, full64[k].bem_pressure, full64[k].bem_neumann,
                                             k64, cache64)
            for i in eachindex(FIELD_POINTS)
                record!(:field_pressure, f, split, k, i, c.field[i, k], pc[i], pr[i])
            end
        end
    end
end

# resonances: local maxima of |diaphragm velocity| (Float64 build, excitation 1, split irrelevant)
vd = Dict(r.f => r.magnitude for r in records if r.output == :diaphragm_velocity && r.exc == 1 && r.split == "auto")
fs = sort(collect(keys(vd)))
resonances = Set(fs[j] for j in 2:(length(fs) - 1) if vd[fs[j]] > vd[fs[j - 1]] && vd[fs[j]] > vd[fs[j + 1]])
arc_max = Dict()
for r in records
    r.output == :field_pressure || continue
    key = (r.f, r.split, r.exc); arc_max[key] = max(get(arc_max, key, 0.0), r.magnitude)
end
cancellation(r) = r.output == :field_pressure && r.magnitude < arc_max[(r.f, r.split, r.exc)] * 10^(-CANCELLATION_DB / 20)
calibration = Set(frequencies[1:2:end])
relative_entries(set) = [r for r in set if !r.zero_ref]
ratio(r) = isnan(r.estimate) ? 0.0 : r.err / r.estimate     # NaN estimate = unknown = hint, never a release
released_wrongly(r, k) = !isnan(r.estimate) && (iszero(r.estimate) ? r.err > 0 : k * r.estimate < r.err)

println("\nresonance frequencies of this sweep (local maxima of |diaphragm velocity|): ", sort(collect(resonances)))
println("calibration frequencies: ", sort(collect(calibration)))
outputs = (:diaphragm_velocity, :voice_coil_current, :interface_velocity, :field_pressure)
const SHORT = Dict(:diaphragm_velocity => "v_diaph", :voice_coil_current => "i_coil", :interface_velocity => "v_iface",
                   :field_pressure => "p_field")
factors = Dict{Symbol,Float64}()
println("\nper output (true error = complex relative error vs the fully Float64 build):")
for o in outputs
    all_o = relative_entries(filter(r -> r.output == o, records))
    cal = filter(r -> r.f in calibration, all_o); hold = filter(r -> !(r.f in calibration), all_o)
    k = max(1.0, maximum(ratio, cal; init=0.0)); factors[o] = k
    @printf("  %-19s entries %3d | max true %.3e | max estimate %.3e | false releases k=1: %d | calibrated k=%.3g on %d entries -> holdout false releases %d of %d (worst holdout true/estimate %.3g)\n",
            o, length(all_o), maximum(r -> r.err, all_o; init=0.0), maximum(r -> isnan(r.estimate) ? 0.0 : r.estimate, all_o; init=0.0),
            count(r -> released_wrongly(r, 1.0), all_o), k, length(cal), count(r -> released_wrongly(r, k), hold),
            length(hold), maximum(ratio, hold; init=0.0))
    zeros = filter(r -> r.output == o && r.zero_ref, records)
    isempty(zeros) || @printf("    exact-zero references: %d, largest absolute error %.3e\n", length(zeros), maximum(r -> r.err, zeros))
end
println("\nseparately: resonances and cancellations (calibrated k per output as above):")
for (name, pick) in (("resonance", r -> r.f in resonances), ("cancellation", cancellation),
                     ("other", r -> !(r.f in resonances) && !cancellation(r)))
    for o in outputs
        sel = relative_entries(filter(r -> r.output == o && pick(r), records))
        isempty(sel) && continue
        worst = sel[argmax([r.err for r in sel])]
        @printf("  %-12s %-19s entries %3d | max true %.3e (%.0f Hz, split=%s, exc %d, #%d, estimate %.3e) | false releases k=1: %d, k=%.3g: %d\n",
                name, o, length(sel), worst.err, worst.f, worst.split, worst.exc, worst.i, worst.estimate,
                count(r -> released_wrongly(r, 1.0), sel), factors[o], count(r -> released_wrongly(r, factors[o]), sel))
    end
end
println("\nper frequency, max true error / max estimate over excitations, splits and entries:")
for f in frequencies
    @printf("  %6.0f Hz%s", f, f in resonances ? " (res)" : "      ")
    for o in outputs
        sel = relative_entries(filter(r -> r.output == o && r.f == f, records))
        @printf(" | %s %.2e / %.2e", SHORT[o], maximum(r -> r.err, sel; init=0.0),
                maximum(r -> isnan(r.estimate) ? 0.0 : r.estimate, sel; init=0.0))
    end
    println()
end
