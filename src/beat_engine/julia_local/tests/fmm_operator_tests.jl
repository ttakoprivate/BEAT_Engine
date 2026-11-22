# FMM Burton-Miller operator: matrix-free product against the dense CPU assembly.
# Needs FMM3D on the load path (it is not a dependency of the engine yet).
isdefined(@__MODULE__, :BeatEngineCore) || include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
isdefined(@__MODULE__, :BeatEngineFmm) || include(joinpath(@__DIR__, "..", "src", "BeatEngineFmm.jl"))
using .BeatEngineFmm
using Test, LinearAlgebra, StaticArrays, Random, SparseArrays

include(joinpath(@__DIR__, "fmm_test_meshes.jl"))

# Half of a sphere centred off the plane x = 0 would not be closed; for the symmetry case
# use a sphere fully in x > 0 and let the image add its mirror (two separate spheres).
function _fmm_case(mesh, k, symmetry; eps=1e-8, order=2, near_factor=2.0)
    p1 = build_p1_space(mesh); dp0 = build_dp0_space(mesh)
    rule = triangle_rule(Float64, order)
    Ipp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
    Ipd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
    q = randn(MersenneTwister(3), ComplexF64, dp0.global_dof_count, 2)
    dense = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, q, k, rule;
        identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=symmetry, singular_order=2)
    op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=Ipp, identity_p1_dp0=Ipd, symmetry_mode=symmetry, eps, near_factor)
    rng = MersenneTwister(7)
    x = randn(rng, ComplexF64, p1.global_dof_count)
    yd = dense.matrix * x
    yf = fmm_lhs_mul(op, x)
    rerr = norm(fmm_rhs(op, q) - dense.rhs) / norm(dense.rhs)
    return max(norm(yf - yd) / norm(yd), rerr), op
end

@testset "FMM Burton-Miller product" begin
    sc, sd = BeatEngineFmm.fmm_kernel_factors()
    @info "FMM3D kernel factors" sc sd
    for (sub, f) in ((2, 1000.0), (3, 3000.0))
        mesh = _fmm_icosphere(0.1, sub)
        k = 2pi * f / 343
        err, _ = _fmm_case(mesh, k, :off)
        @info "sphere" faces = length(mesh.faces) f err
        @test err < 1e-6
    end
    mesh = _fmm_icosphere(0.1, 2; centre=SVector(0.15, 0.0, 0.0))
    err, _ = _fmm_case(mesh, 2pi * 2000 / 343, :x)
    @info "two spheres via x symmetry" err
    @test err < 1e-6
    mesh = _fmm_icosphere(0.1, 2; centre=SVector(0.15, 0.13, 0.0))
    err, _ = _fmm_case(mesh, 2pi * 2000 / 343, :xy)
    @info "four spheres via xy symmetry" err
    @test err < 1e-6
    # a near radius below the element size must not count adjacent pairs twice (review of 3190c88:
    # 17 % boundary-pressure error at near_factor = 0.25 before the fix)
    mesh = _fmm_icosphere(0.1, 1)
    for near_factor in (0.25, 2.0)
        err, _ = _fmm_case(mesh, 2pi * 2000 / 343, :off; near_factor)
        @info "small near radius" near_factor err
        @test err < 1e-6
    end
end

@testset "FmmCoupledSolve reports non-convergence instead of returning the iterate" begin
    mesh = _fmm_icosphere(0.1, 1)
    k = 2pi * 2000 / 343
    _, op = _fmm_case(mesh, k, :off)
    n = op.n_p1; nq = length(mesh.faces)
    none = spzeros(ComplexF64, nq, 0)
    b = fmm_lhs_mul(op, randn(MersenneTwister(7), ComplexF64, n))
    S = FmmCoupledSolve(op, spzeros(ComplexF64, n, n), 1:n, Int[], Int[], none, none; rtol=1e-9, maxiter=1)
    err = try S \ b; nothing catch e; e end
    @test err isa FmmNotConvergedError && err.columns == [1] && err.residuals[1] > 1e-9
    S = FmmCoupledSolve(op, spzeros(ComplexF64, n, n), 1:n, Int[], Int[], none, none; rtol=1e-9, maxiter=300)
    x = S \ b
    @test S.stats[end].solved && norm(fmm_lhs_mul(op, x) - b) <= 1e-8 * norm(b)
end
