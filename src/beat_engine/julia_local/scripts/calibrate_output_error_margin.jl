# Calibration of DENSE_OUTPUT_ROUNDING_MARGIN σ (BeatEngineSplitDense.jl): the smallest σ for which
# the componentwise solver-error bound |d_i| + (ρ + σ)/(1-ρ) ‖d‖ covers the true error of every
# entry (BigFloat reference), on calibration seeds 1:3 and held-out seeds 4:10 of the dense test
# systems in tests/output_error_estimate_tests.jl (κ 1e3..1e7, driver rows scaled 1 and 1e-3).
#
#   julia --project=src/beat_engine/julia_local src/beat_engine/julia_local/scripts/calibrate_output_error_margin.jl
const HERE = joinpath(@__DIR__, "..")
include(joinpath(HERE, "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(HERE, "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(HERE, "tests", "coupled_condensed_test_setup.jl"))
using LinearAlgebra, Random, Printf
const OE = BeatEngineCoupledCondensed
let source = read(joinpath(HERE, "tests", "output_error_estimate_tests.jl"), String)   # oe_fixture, oe_split
    eval(Meta.parse("begin " * source[findfirst("function oe_fixture", source)[1]:findfirst("@testset", source)[1] - 1] * " end"))
end

function required_margin(seeds)
    sigma = OE.BeatEngineCoupled.DENSE_OUTPUT_ROUNDING_MARGIN
    worst = -Inf
    setprecision(BigFloat, 160) do
        for condition in (1e3, 1e5, 1e6, 1e7), border_scale in (1.0, 1e-3), seed in seeds
            rng = MersenneTwister(100 + seed)
            dense, m = oe_fixture(rng, 100, 8; condition, border_scale)
            rhs = randn(rng, ComplexF64, size(dense, 1), 3)
            exact = Complex{BigFloat}.(dense) \ Complex{BigFloat}.(rhs)
            for factorization in (OE.SplitRefinedLU(oe_split(dense, m)), OE.RefinedDenseLU(copy(dense)))
                solution = factorization \ rhs
                bound = factorization.error_bound
                for j in axes(solution, 2)
                    factorization.estimate_direction_check[j] || continue
                    truth = Float64.(abs.(Complex{BigFloat}.(solution[:, j]) .- exact[:, j]))
                    normwise = maximum(bound[:, j]) / (1 + sigma)       # ‖d‖ / (1-ρ)
                    iszero(normwise) && continue
                    worst = max(worst, maximum((truth .- bound[:, j]) ./ normwise) + sigma)
                end
            end
        end
    end
    return worst
end

@printf("required sigma: calibration seeds 1:3 = %.3f, held-out seeds 4:10 = %.3f (constant %.1f)\n",
        required_margin(1:3), required_margin(4:10), OE.BeatEngineCoupled.DENSE_OUTPUT_ROUNDING_MARGIN)
