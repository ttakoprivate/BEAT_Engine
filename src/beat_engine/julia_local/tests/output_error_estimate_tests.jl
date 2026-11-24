# Step 2 of docs/ROADMAP.md (display only): per-output complex relative solver-error estimate from
# the componentwise bound of `_dense_error_estimate`, and the "recompute in Float64" hint.
isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
using Test, LinearAlgebra, Random

const OE = BeatEngineCoupledCondensed

# Dense system like the flux-eliminated build: a Float32-like BEM block plus Float64 border
# rows/columns (the driver unknowns), optionally ill-conditioned and with small border rows.
function oe_fixture(rng, n, border; condition=1e3, border_scale=1.0)
    m = n - border
    bem = if condition > 1e3
        u, _ = qr(randn(rng, ComplexF64, m, m)); v, _ = qr(randn(rng, ComplexF64, m, m))
        ComplexF32.(Matrix(u) * Diagonal(exp10.(range(0, -log10(condition); length=m))) * Matrix(v)')
    else
        randn(rng, ComplexF32, m, m) ./ Float32(sqrt(m)) + 2.0f0I
    end
    dense = zeros(ComplexF64, n, n)
    dense[1:m, 1:m] = bem
    dense[:, (m + 1):n] = randn(rng, ComplexF64, n, border)
    dense[(m + 1):n, 1:m] = border_scale .* randn(rng, ComplexF64, border, m)
    return dense, m
end

function oe_split(dense, m)
    n = size(dense, 1)
    split = OE.SplitDenseMatrix(n, (m + 1):n, (m + 1):n)
    for j in 1:n, i in 1:n
        split[i, j] = dense[i, j]
    end
    return split
end

@testset "componentwise bound covers the true solver error of every entry" begin
    setprecision(BigFloat, 160) do
        smallest = Inf             # smallest bound / truth over entries with a passed check
        for condition in (1e3, 1e5, 1e6), border_scale in (1.0, 1e-3), seed in 1:3
            rng = MersenneTwister(100 + seed)
            dense, m = oe_fixture(rng, 100, 8; condition, border_scale)
            rhs = randn(rng, ComplexF64, size(dense, 1), 3)
            exact = Complex{BigFloat}.(dense) \ Complex{BigFloat}.(rhs)
            for factorization in (OE.SplitRefinedLU(oe_split(dense, m)), OE.RefinedDenseLU(copy(dense)))
                solution = factorization \ rhs
                bound = factorization.error_bound
                @test size(bound) == size(solution)
                rows = (m + 1):size(dense, 1)
                for j in axes(solution, 2)
                    factorization.estimate_direction_check[j] || continue
                    truth = Float64.(abs.(Complex{BigFloat}.(solution[:, j]) .- exact[:, j]))
                    @test all(bound[:, j] .>= truth)
                    positive = truth .> 0
                    any(positive) && (smallest = min(smallest, minimum(bound[positive, j] ./ truth[positive])))
                    relative = OE._output_error_bound(bound[:, j], solution[:, j], rows)
                    @test all(relative .>= truth[rows] ./ abs.(solution[rows, j]))
                end
            end
        end
        @info "smallest componentwise bound / true error" smallest
        @test smallest >= 1
    end
end

@testset "output estimates per field and excitation, diagnostics and hint" begin
    rng = MersenneTwister(7)
    dense, m = oe_fixture(rng, 80, 6)
    rhs = randn(rng, ComplexF64, size(dense, 1), 2)
    for factorization in (OE.SplitRefinedLU(oe_split(dense, m)), OE.RefinedDenseLU(copy(dense)))
        solution = factorization \ rhs
        mechanical = (m + 1):(m + 3); electrical = (m + 4):(m + 6)
        system = (factorization=factorization, mechanical_range=mechanical, electrical_range=electrical,
                  scalar_type=Float32)
        OE._store_output_estimates!(system, solution)
        estimates = factorization.output_estimates
        @test Set(keys(estimates)) == Set([:diaphragm_velocity, :voice_coil_current])
        @test size(estimates[:diaphragm_velocity]) == (3, 2)
        @test all(e -> isfinite(e) && e >= 0, estimates[:voice_coil_current])
        # outputs stored as ComplexF32 carry their rounding √2 u in the estimate
        @test all(>=(sqrt(2) * eps(Float32) / 2), estimates[:diaphragm_velocity])
        OE._store_output_estimates!(merge(system, (scalar_type=Float64,)), solution)
        @test all(factorization.output_estimates[:diaphragm_velocity] .< sqrt(2) * eps(Float32) / 2)
        OE._store_output_estimates!(system, solution)
        estimates = factorization.output_estimates
        diagnostics = withenv("BLAB_OUTPUT_ERROR_HINT" => nothing) do
            OE.dense_solver_diagnostics(system)
        end
        @test diagnostics["dense_output_error_hint_threshold"] == 1e-3
        @test diagnostics["dense_output_estimated_relative_error_diaphragm_velocity"][2] ==
              estimates[:diaphragm_velocity][:, 2]
        # well-conditioned: refined to Float64 level, no hint
        @test diagnostics["dense_output_recompute_float64"] === false
        # a threshold below the estimate raises the hint for that entry
        tiny = minimum(estimates[:voice_coil_current]) / 2
        hinted = withenv("BLAB_OUTPUT_ERROR_HINT" => string(tiny)) do
            OE.dense_solver_diagnostics(system)
        end
        @test hinted["dense_output_recompute_float64"] === true
        @test all(hinted["dense_output_recompute_float64_voice_coil_current"][1])
        @test_throws ArgumentError withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => "-1")
        # a new solve invalidates the stored per-output estimates
        factorization \ rhs
        @test isnothing(factorization.output_estimates)
        @test !haskey(OE.dense_solver_diagnostics(system), "dense_output_recompute_float64")
    end
end

@testset "complex error: equal level with flipped phase raises the hint" begin
    # +1 Pa against -1 Pa: 0 dB level difference, complex relative error 2
    reference = ComplexF64[1, 0.5im]; candidate = -reference
    bound = abs.(candidate .- reference)                 # an exact componentwise bound of that error
    relative = OE._output_error_bound(bound, candidate, eachindex(candidate))
    @test relative ≈ [2.0, 2.0]
    @test all(20 .* log10.(abs.(candidate) ./ abs.(reference)) .== 0)
    @test all(OE.output_recompute_hint.(relative, 1e-3))
    @test all(OE.output_recompute_hint.(relative, 1.0))  # even a lax threshold below 200 %
    # unknown or infinite estimates always ask for the Float64 recompute
    @test OE.output_recompute_hint(NaN, 1.0) && OE.output_recompute_hint(Inf, 1.0)
    @test !OE.output_recompute_hint(1e-6, 1e-3)
    # zero entry: Inf with a non-zero bound, 0 with a zero bound; NaN bound stays unknown
    @test OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[1] == Inf
    @test OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[2] == 0.0
    @test isnan(OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[3])
end

@testset "failed direction check leaves the componentwise bound unknown and raises the hint" begin
    # review case 2 of PR #2: a Float32 pivot shifted by rounding, the direction check fails
    for tail in (2.0^-28, 2.0^-34)
        a = 1 + 2.0^-24
        A = ComplexF64[1 a; 1 (a + tail)]
        b = reshape((1 + im) .* [1, 1 + eps()], 2, 1)
        for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
            x = factorization \ b
            @test !factorization.estimate_direction_check[1]
            @test all(isnan, factorization.error_bound[:, 1])
            relative = OE._output_error_bound(factorization.error_bound[:, 1], x[:, 1], 1:2)
            @test all(OE.output_recompute_hint.(relative, 1.0))
        end
    end
end

@testset "REVIEW-22-b5ea266 P1: residual rounding is bounded, not taken from the correction" begin
    # nearly singular 2x2 with a scaled second row; the residual of the converged solve is pure
    # rounding, so the former σ ‖d‖ term underestimated entry 2 by about 90x without a hint
    rng = MersenneTwister(1)
    row = randn(rng, ComplexF64, 2)
    A = vcat(transpose(row), transpose(row + 1e-4 * randn(rng, ComplexF64, 2)))
    A[2, :] .*= 1e-3
    b = reshape(A[:, 1] + 7e-12 * A[:, 2], 2, 1)
    exact = setprecision(BigFloat, 256) do
        Complex{BigFloat}.(A) \ Complex{BigFloat}.(b)
    end
    for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
        x = factorization \ b
        truth = Float64.(abs.(Complex{BigFloat}.(x[:, 1]) .- exact[:, 1]))
        @test isnothing(factorization.fallback)
        @test all(b -> isnan(b) || b >= 0, factorization.error_bound)
        @test all(isnan.(factorization.error_bound[:, 1]) .| (factorization.error_bound[:, 1] .>= truth))
        system = (factorization=factorization, mechanical_range=2:2, electrical_range=1:1, scalar_type=Float32)
        OE._store_output_estimates!(system, x)
        diagnostics = withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => nothing)
        # the true relative error of entry 2 is several percent: the hint must be raised
        @test truth[2] / abs(x[2, 1]) > 1e-3
        @test diagnostics["dense_output_recompute_float64_diaphragm_velocity"][1][1] === true
        @test diagnostics["dense_output_recompute_float64"] === true
    end
end

@testset "REVIEW-22-b5ea266 P1: a residual rounded to zero does not give a zero bound" begin
    A = reshape(ComplexF64[3], 1, 1); b = reshape(ComplexF64[1], 1, 1)
    exact = setprecision(BigFloat, 256) do
        Complex{BigFloat}.(A) \ Complex{BigFloat}.(b)
    end
    for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
        x = factorization \ b
        truth = Float64(abs(Complex{BigFloat}(x[1, 1]) - exact[1, 1]))
        @test truth > 0
        @test factorization.error_bound[1, 1] >= truth
    end
    # the estimator alone, with the reviewer's x = Float64(1/3) and its rounded-to-zero residual
    x = reshape(ComplexF64[1 / 3], 1, 1)
    residual = b .- A * x
    @test iszero(residual)
    rounding = OE._residual_rounding(b, abs.(A) * abs.(x), 1)
    _, passed, bound = OE._dense_error_estimate(r -> A \ r, y -> A * y, x, residual, b;
                                                apply_adjoint=r -> A' \ r, rounding)
    @test passed == [true]
    @test bound[1, 1] >= Float64(abs(big(1) / 3 - big(1 / 3)))
    # without an adjoint solve (device factor) the componentwise bound is unknown
    _, _, unknown = OE._dense_error_estimate(r -> A \ r, y -> A * y, x, residual, b)
    @test isnan(unknown[1, 1])
end

@testset "REVIEW-22-b5ea266 P2: SplitGMRES without fallback reports unknown outputs and the hint" begin
    A = ComplexF64[1 0; 0 1]
    solver = OE.SplitGMRES(oe_split(A, 0))
    x = solver \ reshape(ComplexF64[1, 2], 2, 1)
    @test isnothing(solver.fallback)
    system = (factorization=solver, mechanical_range=2:2, electrical_range=1:1, scalar_type=Float32)
    OE._store_output_estimates!(system, x)
    diagnostics = withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => nothing)
    @test diagnostics["dense_output_estimated_relative_error_diaphragm_velocity"] == [[nothing]]
    @test diagnostics["dense_output_estimated_relative_error_voice_coil_current"] == [[nothing]]
    @test diagnostics["dense_output_recompute_float64_diaphragm_velocity"] == [[true]]
    @test diagnostics["dense_output_recompute_float64"] === true
    solver \ reshape(ComplexF64[1, 2], 2, 1)          # a new solve invalidates them
    @test !haskey(OE.dense_solver_diagnostics(system), "dense_output_recompute_float64")
end
