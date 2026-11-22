# check_truth (dump checker) acceptance tests from the reviews of PR #3; CPU only:
#   julia --project=. hmatrix_dev/tests/hmat_check_truth_test.jl     (from src/beat_engine/julia_local)
using Test, LinearAlgebra, SparseArrays, StaticArrays
include(joinpath(@__DIR__, "..", "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
const BoundaryMesh = BeatEngineCore.BoundaryMesh
let src = read(joinpath(@__DIR__, "hmat_dump_check.jl"), String)    # only the function, not the script
    i = findfirst("function check_truth", src)[1]; j = findnext("\nend\n", src, i)[end]
    eval(Meta.parse(src[i:j]))
end

V = [SVector(0.0, 0, 0), SVector(1.0, 0, 0), SVector(0.0, 1, 0)]
mesh(v) = BoundaryMesh(v, [(1, 2, 3)], [7])
base = (model_scalar_type=:Float64, bem_backend=:cpu, assembly_eltype=:ComplexF64, solver_scalar_type=:Float64,
        solution_eltype=:ComplexF64, dense_solver=:lu_float64, frequency_hz=2000.0, wavenumber=2pi * 2000 / 343, rest=sparse([1.0+0im 0; 0 1]),
        bem_flux=sparse(ComplexF64[1 0; 0 1]), rhs=ComplexF64[1 0; 0 1], bem_mesh=mesh(V),
        regular_quadrature_order=1, singular_order=1, symmetry_mode=:none)
accepted(t, d) = try check_truth(t, d); true catch; false end
label(t, d) = try check_truth(t, d) catch; "rejected" end

@testset "check_truth" begin
    @test accepted(base, base)
    V32 = [SVector{3,Float64}(Float32.(v .+ 0.1)) .- 0.1 for v in V]            # same input, Float32 rounding
    @test accepted(base, merge(base, (; bem_mesh=mesh(V32))))
    @test !accepted(base, merge(base, (; bem_mesh=mesh([V[1], V[2] + SVector(1e-3, 0, 0), V[3]]))))  # moved node
    @test !accepted(base, merge(base, (; bem_mesh=BoundaryMesh(V, [(1, 3, 2)], [7]))))              # topology
    @test !accepted(base, merge(base, (; bem_mesh=BoundaryMesh(V, [(1, 2, 3)], [8]))))              # tags
    @test !accepted(merge(base, (; bem_flux=sparse(ComplexF64[1 0; 0 1; 0 0]))), base)              # extra zero row
    @test !accepted(merge(base, (; rest=sparse([2.0+0im 0; 0 1]))), base)                           # e.g. coil R
    @test !accepted(merge(base, (; rhs=ComplexF64[0 1; 1 0])), base)                                # swapped
    @test !accepted(merge(base, (; dense_solver=:lu_float32_refined)), base)
    @test !accepted(merge(base, (; model_scalar_type=:Float32)), base)
    @test !accepted(merge(base, (; bem_backend=:metal)), base)
    # medium: same frequency, different sound speed (review of a633e67); Float32 rounding of the same k is accepted
    @test accepted(base, merge(base, (; wavenumber=Float64(Float32(2pi * 2000 / 343)))))
    @test accepted(base, merge(base, (; wavenumber=2pi * 2000 / Float64(Float32(343.0)) * (1 + eps(Float32)))))
    @test !accepted(base, merge(base, (; wavenumber=2pi * 2000 / 330)))
    @test !accepted(base, merge(base, (; wavenumber=2pi * 2000 / 342)))
    @test !accepted(base, merge(base, (; wavenumber=2pi * 2000 / 342.99)))
    # invalid wavenumbers (review of f63a9b1): NaN/Inf compare false and must not pass
    for bad in (NaN, Inf, -Inf, 0.0, -2pi * 2000 / 343)
        @test !accepted(merge(base, (; wavenumber=bad)), base)
        @test !accepted(base, merge(base, (; wavenumber=bad)))
    end
    # discretisation (review of f63a9b1): regular/singular quadrature order and symmetry must match
    @test !accepted(merge(base, (; regular_quadrature_order=2)), base)
    @test !accepted(merge(base, (; singular_order=2)), base)
    @test !accepted(merge(base, (; symmetry_mode=:xz)), base)
    @test !accepted(Base.structdiff(base, NamedTuple{(:regular_quadrature_order,)}), base)   # missing in truth
    withenv("HMAT_TRUTH_DISCRETISATION_STUDY" => "1") do      # deliberate quadrature study: own label
        @test startswith(label(merge(base, (; regular_quadrature_order=2)), base), "Float64 discretisation-study reference")
        @test label(base, base) == "verified Float64 dump"
        @test !accepted(merge(base, (; regular_quadrature_order=2, wavenumber=NaN)), base)  # other checks still apply
    end
    withenv("HMAT_TRUTH_UNCHECKED" => "1") do
        @test label(merge(base, (; singular_order=2)), base) == "UNCHECKED reference"
    end
end
