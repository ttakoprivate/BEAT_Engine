# ACA regression tests from the review of 3190c88 (CPU, no GPU needed):
#   julia --project=../.. hmatrix_dev/tests/hmat_aca_test.jl     (from src/beat_engine/julia_local)
using Test, LinearAlgebra
# repository layout: hmatrix_dev/tests/ next to hmatrix_dev/*.jl; Mac kit layout: all files in one folder
const HMAT_DIR = isfile(joinpath(@__DIR__, "hmat_core.jl")) ? (@__DIR__) : joinpath(@__DIR__, "..")
include(joinpath(HMAT_DIR, "hmat_core.jl"))

aca_of(A, tol; kw...) = aca(i -> A[i, :], j -> A[:, j], size(A)..., tol; kw...)

@testset "ACA: relative stop independent of the block scale" begin
    for s in (1.0, 1e-16, 1e-150, 1e-170, 1e160)
        A = s * Matrix(Diagonal(ComplexF64[1 + im, 1 - im]))
        U, V = aca_of(A, 1e-5)
        x = ones(ComplexF64, 2)
        @test size(U, 2) == 2
        @test norm(U * (V' * x) - A * x) <= 1e-12 * norm(A * x)
    end
end

@testset "ACA: an empty start row does not consume the rank budget" begin
    A = reshape(ComplexF64[0, 1 + im, 2 - im], 3, 1)
    U, V = aca_of(A, 1e-5; maxrank=1)
    @test size(U, 2) == 1
    @test norm(U * V' - A) <= 1e-12 * norm(A)
end

@testset "ACA: low-rank block to tolerance" begin
    x = range(0, 1; length=60); y = range(3, 4; length=50)
    A = ComplexF64[cis(2pi * xi * yj) / (yj - xi) for xi in x, yj in y]
    U, V = aca_of(A, 1e-6)
    @test norm(U * V' - A) <= 1e-5 * norm(A)
end

# batched ACA (hmat_ka.jl) with the same cases; needs KernelAbstractions/Adapt on the load path
if Base.find_package("KernelAbstractions") !== nothing && Base.find_package("Adapt") !== nothing
    using SparseArrays, Printf, Random
    include(joinpath(HMAT_DIR, "hmat_ka.jl"))
    batched_of(A, tol) = begin
        st = [ACAState(collect(1:size(A, 1)), collect(1:size(A, 2)), false)]
        aca_batched(st, (EI, EJ, need) -> ComplexF64[A[EI[m], EJ[m]] for m in eachindex(EI)], tol)
        s = st[1]
        (isempty(s.U) ? zeros(ComplexF64, size(A, 1), 0) : reduce(hcat, s.U),
         isempty(s.V) ? zeros(ComplexF64, size(A, 2), 0) : reduce(hcat, s.V))
    end
    @testset "batched ACA: relative stop independent of the block scale" begin
        for s in (1.0, 1e-16, 1e-170, 1e160)
            A = s * Matrix(Diagonal(ComplexF64[1 + im, 1 - im]))
            U, V = batched_of(A, 1e-5)
            @test norm(U * (V' * ones(2)) - A * ones(2)) <= 1e-12 * norm(A * ones(2))
        end
        A = reshape(ComplexF64[0, 1 + im, 2 - im], 3, 1)
        U, V = batched_of(A, 1e-5)
        @test norm(U * V' - A) <= 1e-12 * norm(A)
    end
else
    @info "KernelAbstractions not on the load path: batched ACA tests skipped"
end
