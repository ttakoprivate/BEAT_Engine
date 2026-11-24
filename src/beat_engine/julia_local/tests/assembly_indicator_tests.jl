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

@testset "Driver-only stratification" begin
    n = 12
    sys = (bem_range=1:8, mechanical_range=9:10, electrical_range=11:12,
           factorization=(matrix=Matrix{ComplexF64}(I,n,n),))
    state = (x=ones(ComplexF64,n), rhs=zeros(ComplexF64,n))
    adjoints = zeros(ComplexF64,n,4)
    adjoints[2,1] = 1
    adjoints[7,4] = 1e-8im
    draw(budget) = AIE.driver_sample(MersenneTwister(3), sys, state, adjoints, budget)
    sample = draw(5)
    @test sort(sample.rows[1:2]) == [2,7]
    @test sample.probability[1:2] == [1,1]
    @test sample.probability[3:5] == fill(0.5,3)
    @test sample.rows[end-3:end] == collect(9:12)
    @test sample == draw(5)
    @test draw(1).probability[1] == 1/8
    @test draw(100).rows == collect(1:n)
    @test all(isone, draw(8).probability)
    @test_throws ArgumentError draw(0)
end

@testset "Explicit sample validation" begin
    dual = ones(ComplexF64, 4)
    residual = ComplexF64[1, 1, 1, 9]
    valid = (rows=collect(1:4), probability=fill(0.5, 4), groups=[1:4])
    @test AIE.project_output(valid, dual, residual, 0).stderr > 0
    for groups in ([1:2], [1:3, 3:4], [1:4, 1:0], [0:3], [2:5])
        @test_throws ArgumentError AIE.project_output(merge(valid, (; groups)), dual, residual, 0)
    end
    unequal = merge(valid, (probability=[0.5, 0.5, 0.25, 0.25],))
    @test_throws ArgumentError AIE.project_output(unequal, dual, residual, 0)
    @test_throws DimensionMismatch AIE.project_output(valid, dual, [1], 0)
    @test_throws DimensionMismatch AIE.project_output(valid, dual, ones(5), 0)
    @test_throws DimensionMismatch AIE.project_output(merge(valid, (probability=[0.5],)), dual, residual, 0)
    for rows in ([1, 1, 3, 4], [0, 2, 3, 4], [1, 2, 3, 5])
        @test_throws ArgumentError AIE.project_output(merge(valid, (; rows)), dual, residual, 0)
    end
    for probability in ([NaN, 0.5, 0.5, 0.5], zeros(4), fill(1.1, 4))
        @test_throws ArgumentError AIE.project_output(merge(valid, (; probability)), dual, residual, 0)
    end
    census = merge(valid, (probability=ones(4), groups=[1:2, 3:4]))
    @test AIE.project_output(census, dual, residual, 0).delta == sum(residual)
    @test iszero(AIE.project_output(census, dual, residual, 0).stderr)
    singleton = (rows=[1], probability=[0.5], groups=[1:1])
    @test isinf(AIE.project_output(singleton, dual, [1], 0).stderr)
    @test_throws ArgumentError AIE.project_output((rows=Int[], probability=Float64[], groups=UnitRange{Int}[]), dual, ComplexF64[], 0)
    for budget in (true, 2.0)
        @test_throws ArgumentError sample_equations(MersenneTwister(0), 4, (1:4,), (budget,))
    end
end
