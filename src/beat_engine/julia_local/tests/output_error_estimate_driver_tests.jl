# Roadmap step 2 (display only): the per-output and per-point error estimates through the real JSON
# driver. Until now the driver wiring of field_output_error_diagnostics! was only parsed, never run.
# The tiny compiled coupled workload and the 2-kHz fixture (femvolume + exterior_conforming) are solved
# through solve_request(...; event_mode=true) in Float32 (default CPU path) and Float64. Checked:
# diagnostic keys and shapes per output, excitation and point, consistency of the Float64 hints with
# the threshold (BLAB_OUTPUT_ERROR_HINT), the synthesized path (excitation_weights, phase inversion),
# and that a Float64 request reports no estimate. The maximum complex relative error of every output
# against the Float64 driver run is printed next to the estimate. It is not asserted as a bound: the
# Float64 run also assembles in Float64, so the difference contains the Float32 assembly error that
# this estimate does not cover (roadmap step 4); the automatic release stays off.
module OutputErrorEstimateDriverTests
using Test
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

decode(q) = ComplexF64.(copy(reinterpret(q["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64,
                                         base64decode(q["values"]["content_base64"]))))
quantity(result, id) = only(q for q in result["quantities"] if q["id"] == id)

const FIELD = "ui:exterior-pressure"
const SCALARS = ("diaphragm_velocity" => "mechanical:diaphragm-velocity",
                 "voice_coil_current" => "electrical:voice-coil-current")
const INTERFACE = "acoustic:interface-average-normal-velocity"

function workload(; tiny::Bool, precision::String="float32")
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny)))
    tiny || (request["frequencies_hz"] = [2000.0])
    request["solver_options"]["precision"] = precision
    return request
end

run_workload(request; env=()) = withenv(coupled_workload_environment(; mumps=false, defaults=:cpu)..., env...) do
    solve_coupled_workload(request)
end

# Shape [excitation][point] with finite, positive entries; hints follow the threshold.
function check_estimates(d; hinted_everywhere::Bool=false)
    threshold = d["dense_output_error_hint_threshold"]
    @test threshold > 0
    field = d["dense_output_estimated_relative_error_field"][FIELD]
    hints = d["dense_output_recompute_float64_field"][FIELD]
    @test length(field) == 1 && length(only(field)) == 2
    @test all(v -> v isa Real && isfinite(v) && v > 0, only(field))
    @test hints == [[!(v <= threshold) for v in only(field)]]
    @test d["dense_output_field_evaluation_rounding_included"][FIELD] == true
    @test !haskey(d, "dense_output_field_error_not_computed")
    @test d["dense_output_field_error_s"] >= 0
    any_hint = any(only(hints))
    @test d["dense_output_recompute_float64_fields"] == any_hint
    for (name, _) in SCALARS
        values = d["dense_output_estimated_relative_error_$(name)"]
        @test length(values) == 1 && length(only(values)) == 1
        @test all(v -> isfinite(v) && v > 0, only(values))
        @test d["dense_output_recompute_float64_$(name)"] == [[!(v <= threshold) for v in only(values)]]
        any_hint |= any(only(d["dense_output_recompute_float64_$(name)"]))
    end
    # Interface average normal velocity: [excitation][interface], finite and positive.
    velocity = d["dense_output_estimated_relative_error_interface_velocity"][INTERFACE]
    velocity_hints = d["dense_output_recompute_float64_interface_velocity"][INTERFACE]
    @test length(velocity) == 1 && length(only(velocity)) == 1
    @test all(v -> v isa Real && isfinite(v) && v > 0, only(velocity))
    @test velocity_hints == [[!(v <= threshold) for v in only(velocity)]]
    @test d["dense_output_recompute_float64_interface_velocities"] == any(only(velocity_hints))
    any_hint |= any(only(velocity_hints))
    @test d["dense_output_recompute_float64"] == any_hint
    hinted_everywhere && @test any_hint && all(only(hints)) && all(only(velocity_hints))
    return field
end

max_relative(x, y) = maximum(abs.(x .- y) ./ abs.(y))

@testset "Output error estimates through the JSON driver (roadmap step 2)" begin
    for tiny in (true, false)
        label = tiny ? "tiny" : "2-kHz fixture"
        r32 = run_workload(workload(; tiny))
        r64 = run_workload(workload(; tiny, precision="float64"))
        nfreq = tiny ? 2 : 1
        @test r32.outcome.solved_count == nfreq && r64.outcome.solved_count == nfreq
        for (a, b) in zip(r32.results, r64.results)
            d = a["diagnostics"]
            @test startswith(d["dense_solver"], "lu_float32")
            check_estimates(d)
            # The Float64 request has no Float32 solve to estimate.
            @test !any(startswith(k, "dense_output_estimated") for k in keys(b["diagnostics"]))
            # All reference values are nonzero here, so the relative error is defined everywhere.
            @test all(!iszero, decode(quantity(b, FIELD)))
            field_error = abs.(decode(quantity(a, FIELD)) .- decode(quantity(b, FIELD))) ./
                          abs.(decode(quantity(b, FIELD)))
            println("  $label $(a["freq_hz"]) Hz, 1 excitation: exterior pressure (Pa) per point ",
                    "max complex rel. error vs Float64 driver run = ", field_error,
                    ", estimate = ", only(d["dense_output_estimated_relative_error_field"][FIELD]))
            for (name, id) in SCALARS
                println("  $label $(a["freq_hz"]) Hz, 1 excitation: $id max complex rel. error = ",
                        max_relative(decode(quantity(a, id)), decode(quantity(b, id))),
                        ", estimate = ", only(only(d["dense_output_estimated_relative_error_$(name)"])))
            end
            println("  $label $(a["freq_hz"]) Hz, 1 excitation: $INTERFACE max complex rel. error = ",
                    max_relative(decode(quantity(a, INTERFACE)), decode(quantity(b, INTERFACE))), ", estimate = ",
                    only(only(d["dense_output_estimated_relative_error_interface_velocity"][INTERFACE])))
        end
    end
end

@testset "Synthesized field estimate and threshold through the driver (roadmap step 2)" begin
    plain = run_workload(workload(; tiny=true))
    request = workload(; tiny=true)
    output = only(o for o in request["outputs"] if o["id"] == FIELD)
    # Phase inversion: same level, complex value negated.
    output["options"]["excitation_weights"] = [Dict("real" => -1.0, "imag" => 0.0)]
    inverted = run_workload(request)
    @test inverted.outcome.solved_count == 2
    for (a, b) in zip(plain.results, inverted.results)
        d = b["diagnostics"]
        @test d["dense_output_field_error_synthesized"][FIELD] == 1
        estimate = check_estimates(d)
        @test decode(quantity(b, FIELD)) == -decode(quantity(a, FIELD))
        # The negated field has exactly the same error as the plain one. The synthesized estimate adds
        # the rounding of the weighted combination, so it may be larger (about 10 % here), never smaller,
        # and stays of the same order.
        plain_estimate = only(a["diagnostics"]["dense_output_estimated_relative_error_field"][FIELD])
        @test all(only(estimate) .>= plain_estimate) && all(only(estimate) .<= 2 .* plain_estimate)
    end
    tight = run_workload(workload(; tiny=true); env=("BLAB_OUTPUT_ERROR_HINT" => "1e-12",))
    for result in tight.results
        d = result["diagnostics"]
        @test d["dense_output_error_hint_threshold"] == 1e-12
        check_estimates(d; hinted_everywhere=true)
        @test d["dense_output_recompute_float64_fields"] == true
    end
end
end
