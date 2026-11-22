using Test, LinearAlgebra, Random
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupledCondensed.jl"))
include(joinpath(@__DIR__, "..", "src", "BeatEngineAssemblyIndicator.jl"))
using .BeatEngineAssemblyIndicator
const AIE = BeatEngineAssemblyIndicator

@testset "Independent sampled complex residual" begin
    strata = ([4, 2], [3, 1], Int[])
    draw = sample_equations(MersenneTwister(17), 4, strata, (1, 2, 0))
    @test draw == sample_equations(MersenneTwister(17), 4, strata, (1, 2, 0))
    @test draw.probability == [0.5, 1, 1]
    @test sort(draw.rows[2:3]) == [1, 3]
    @test_throws ArgumentError sample_equations(MersenneTwister(0), 3, ([1, 2], [2, 3]), (1, 1))
    @test_throws ArgumentError sample_equations(MersenneTwister(0), 3, (1:3,), (0,))
    @test_throws DimensionMismatch sample_equations(MersenneTwister(0), 3, (1:3,), (1, 2))

    a = ComplexF64[2+im 0; 1 3-im]
    b = ComplexF64[1-im, 2+im]
    g = ComplexF64[2+im, -4im]
    db = ComplexF64[-1im, 3+im] .* 1e-5
    dg = ComplexF64[0, 1im] .* 1e-5
    x = a \ b
    z = a' \ g
    direct = dot(dg, x)
    expected = dot(g, a \ db) + direct
    census = sample_equations(MersenneTwister(0), 2, (1:2,), (2,))
    p = AIE.project_output(census, z, db[census.rows], direct)
    @test p.delta ≈ expected
    @test iszero(p.stderr)
    # Enumerating the two equally probable single-row draws gives the exact projection.
    draws = [(rows=[i], probability=[0.5], groups=[1:1]) for i in 1:2]
    projected = [AIE.project_output(s, z, db[s.rows], direct) for s in draws]
    @test sum(y.delta for y in projected)/2 ≈ expected
    @test all(y -> isinf(y.stderr), projected)
    silent = AIE.project_output(first(draws), z, ComplexF64[0], 0)
    @test iszero(silent.delta) && isinf(silent.stderr)
    # The unbiased finite-population variance can be verified by enumerating all pairs.
    terms = ComplexF64[1+im, -2im, 4]
    pairs = [[1,2], [1,3], [2,3]]
    pairsums = [AIE.project_output((rows=rows, probability=fill(2/3, 2), groups=[1:2]),
                                  ones(ComplexF64, 3), terms[rows], 0) for rows in pairs]
    @test sum(y.stderr^2 for y in pairsums)/3 ≈ sum(abs2(y.delta-sum(terms)) for y in pairsums)/3
    @test AIE.relative_change(0, 0) == 0
    @test isinf(AIE.relative_change(1im, 0))
    @test AIE.relative_change(2im, 1im) == 2
    @test_throws DomainError AIE.relative_change(1, NaN)
end
