# Regression for review REVIEW-22-89238e4 (P2): the calibration/holdout protocol must compare the Float32
# field evaluation that is actually handed to the estimate, not a Float64 re-evaluation of the Float32
# boundary data; and "under-coverage" of the bound is counted apart from wrong threshold decisions.
# Runs the unchanged protocol script for one frequency (CPU, a few seconds to minutes).
#
#   julia --project=src/beat_engine/julia_local src/beat_engine/julia_local/tests/qualify_output_error_estimate_tests.jl
using Test
empty!(ARGS); push!(ARGS, "1000")
include(joinpath(@__DIR__, "..", "scripts", "qualify_output_error_estimate.jl"))

@testset "qualification protocol compares the actual Float32 field candidate" begin
    field = filter(r -> r.output == :field_pressure, records)
    reeval = filter(r -> r.output == :field_pressure_f64_reeval, records)
    @test length(field) == length(reeval) == length(FIELD_POINTS) * length(SPLITS) * 2
    for r in field
        cand = candidate_fields[(r.f, r.split, r.exc)]
        ref = reference_fields[(r.f, r.split, r.exc)]
        @test eltype(cand) == ComplexF32                      # the Float32 evaluation, not a re-evaluation
        @test r.err == abs(ComplexF64(cand[r.i]) - ref[r.i]) / abs(ref[r.i])
        @test !r.zero_ref
    end
    # the diagnostic is a different path (field rounding removed) and kept under its own name
    @test any(((a, b),) -> a.err != b.err, zip(field, reeval))
    @test all(((a, b),) -> a.estimate == b.estimate && a.magnitude == b.magnitude, zip(field, reeval))
end

@testset "under-coverage is not a wrong release at a threshold" begin
    r = (; estimate=1e-8, err=2e-8)
    @test under_covered(r, 1.0)
    @test !wrong_release(r, 1.0, 1e-4)                      # both below the threshold: decision is right
    @test wrong_release((; estimate=1e-5, err=2e-4), 1.0, 1e-4)
    @test !wrong_release((; estimate=1e-5, err=2e-4), 20.0, 1e-4)
    @test !wrong_release((; estimate=NaN, err=1.0), 1.0, 1e-4)  # unknown estimate = hint, never a release
    @test under_covered((; estimate=0.0, err=1e-12), 1e9)
    @test UNIT[:field_pressure] == "Pa" && UNIT[:voice_coil_current] == "A"
end
