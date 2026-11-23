# Independent CPU checks for the experimental H-matrix branch (issue #11).
# From src/beat_engine/julia_local:
#   OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 julia --project=. --threads=2 tests/hmatrix_reference/runtests.jl
# The prototype needs FMM3D, Krylov, LinearMaps, KernelAbstractions (tested with 0.9.44)
# and Adapt on JULIA_LOAD_PATH.
# To check a pending H-matrix PR, set --project to that checkout's julia_local and run this file.
using Test, LinearAlgebra, SparseArrays, StaticArrays, Random, Printf
const LOCAL = dirname(Base.active_project())
const HM = joinpath(LOCAL, "hmatrix_dev")
include(joinpath(LOCAL, "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(LOCAL, "src", "BeatEngineCoupled.jl"))
include(joinpath(LOCAL, "tests", "coupled_condensed_test_setup.jl"))
include(joinpath(LOCAL, "src", "BeatEngineFmm.jl"))
using .BeatEngineFmm
const BC = BeatEngineCore
using Krylov, LinearMaps
for file in ("hmat_core.jl", "hmat_fast.jl", "hmat_ka.jl", "hmat_c.jl")
    include(joinpath(HM, file))
end
include(joinpath(LOCAL, "tests", "fmm_test_meshes.jl"))

# Never divide by a zero reference or discard quiet components with a level cutoff.
function errors(actual, reference)
    axes(actual) == axes(reference) || throw(DimensionMismatch("different outputs"))
    relative, absolute, zeros = 0.0, 0.0, 0
    for (x, r) in zip(actual, reference)
        if iszero(r)
            absolute = max(absolute, abs(x))
            zeros += 1
        else
            relative = max(relative, abs(x - r) / abs(r))
        end
    end
    return (; relative, absolute, zeros)
end

function build_h_operators(mesh, op, p1, dp0, rule, symmetry, faces, tol, leaf)
    oracle = FastOracle(mesh, p1, dp0, op.k, rule, op, symmetry)
    geometry = ka_geom(oracle, Float64, identity)
    points = reduce(hcat, mesh.vertices)
    centres = reduce(hcat, mesh.centroids[faces])
    root = hcluster(points, collect(axes(points, 2)), leaf)
    cols = hcluster(centres, collect(axes(centres, 2)), leaf)
    images = BC.symmetry_image_transforms(BC.normalized_symmetry_mode(symmetry))
    # Use all image boxes as well as the identity box when classifying a pair.
    function admissible(a, b)
        distance = hdist(a, b)
        for image in images
            u, v = BC.reflect_point(image, SVector{3}(b.lo)), BC.reflect_point(image, SVector{3}(b.hi))
            box = HCluster(b.idx, min.(u, v), max.(u, v), HCluster[])
            distance = min(distance, hdist(a, box))
        end
        return distance > 0 && min(hdiam(a), hdiam(b)) <= distance
    end
    ha, _, _ = hbuild_gpu(oracle, geometry, identity, root, admissible, tol, length(mesh.vertices))
    hc = hbuild_c_gpu(oracle, op, geometry, identity, root, cols, faces, admissible, tol)
    return ha, hc
end

function check_blocks(h, dense, x, tol)
    @test !isempty(h.dense) && !isempty(h.lr)
    coverage = zeros(Int, size(dense))
    for (blocks, limit) in ((h.dense, 1e-8), (h.lr, 50 * tol))
        worst = 0.0
        null_error = 0.0
        for blk in blocks
            r, c = blk.rows, blk.cols
            coverage[r, c] .+= 1
            product = blk isa DenseBlk ? blk.M * x[c, :] : blk.U * (blk.V' * x[c, :])
            e = errors(product, dense[r, c] * x[c, :])
            worst, null_error = max(worst, e.relative), max(null_error, e.absolute)
        end
        @test worst <= limit
        @test null_error <= 1e-13
    end
    @test all(isone, coverage)
    for v in eachcol(x)
        e = errors(hmul(h, v), dense * v)
        @test e.relative <= 50 * tol && e.absolute <= 1e-13
    end
end

function test_blocks_and_krylov()
    mesh = _fmm_icosphere(0.08, 1; centre=SVector(0.08, 0.10, 0.03))
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    n, nf = p1.global_dof_count, dp0.global_dof_count
    rule = triangle_rule(Float64, 2)
    k = 2pi * 2000 / 343
    # Reorder the face basis and apply genuinely different complex excitations.
    faces = collect(nf:-1:1)
    rng = MersenneTwister(37)
    u, q = randn(rng, ComplexF64, n, 2), randn(rng, ComplexF64, nf, 2)
    for symmetry in (:off, :xy)
        pp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1; symmetry_mode=symmetry)
        pd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0; symmetry_mode=symmetry)
        identity_flux = Matrix{ComplexF64}(I, nf, nf)
        dense = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, identity_flux, k, rule;
                    identity_p1_p1=pp, identity_p1_dp0=pd, singular_order=2, symmetry_mode=symmetry)
        for near in (0.25, 4.0)
            op = fmm_burton_miller_operator(mesh, p1, dp0, k, rule; identity_p1_p1=pp,
                    identity_p1_dp0=pd, singular_order=2, symmetry_mode=symmetry, near_factor=near, eps=1e-11)
            dofs = [op.elements[i].dp0_dof for i in faces]
            @test errors(fmm_lhs_mul(op, u[:, 1]), dense.matrix * u[:, 1]).relative <= 1e-6
            @test errors(fmm_rhs(op, q), dense.rhs * q).relative <= 1e-6
            for tol in (1e-3, 1e-4, 1e-5, 1e-6)
                @testset "H products $symmetry, near=$near, tol=$tol" begin
                    a, c = build_h_operators(mesh, op, p1, dp0, rule, symmetry, faces, tol, 5)
                    check_blocks(a, dense.matrix, u, tol)
                    check_blocks(c, dense.rhs[:, dofs], q[dofs, :], tol)
                end
            end
            if symmetry === :off && near == 0.25
                none = spzeros(ComplexF64, nf, 0)
                make_solver(budget) = FmmCoupledSolve(op, spzeros(ComplexF64, n, n), 1:n, Int[], Int[],
                                                      none, none; rtol=1e-10, maxiter=budget)
                b = hcat(zeros(ComplexF64, n), dense.matrix * u[:, 1])
                failure = try
                    make_solver(1) \ b
                    nothing
                catch err
                    err
                end
                @test isdefined(BeatEngineFmm, :FmmNotConvergedError) && failure isa BeatEngineFmm.FmmNotConvergedError
                @test hasproperty(failure, :columns)
                @test hasproperty(failure, :residuals)
                if hasproperty(failure, :columns) && hasproperty(failure, :residuals)
                    @test failure.columns == [2] && only(failure.residuals) > 1e-10
                end
                answer = make_solver(150) \ b
                @test all(iszero, answer[:, 1])
                @test errors(answer[:, 2], dense.matrix \ b[:, 2]).relative < 1e-6
            end
        end
    end
end

function test_aca()
    # The first candidate row can be empty. All nonzero columns span the same two-dimensional space.
    a = ComplexF64[0 0 0; 1+im 0 1; 0 2-im im; 1+im 2-im 1+im]
    rhs = ComplexF64[1 im; -im 2; 2-im -1]
    for factor in (2.0^-550, 1.0, 2.0^530), tol in (1e-3, 1e-6)
        ref = factor .* a
        u, v = aca(i -> ref[i, :], j -> ref[:, j], size(a)..., tol; maxrank=2)
        state = ACAState(collect(axes(a, 1)), collect(axes(a, 2)), false)
        aca_batched([state], (ii, jj, _) -> [ref[i, j] for (i,j) in zip(ii,jj)], tol)
        scalar = u * (v' * rhs)
        batch = isempty(state.U) ? zeros(ComplexF64, size(scalar)) :
                reduce(hcat, state.U) * (reduce(hcat, state.V)' * rhs)
        for result in (scalar, batch)
            e = errors(result, ref * rhs)
            @test e.relative <= 1e-12 && iszero(e.absolute)
        end
    end
    for small in (2.0^-700, 2.0^520)
        a = ComplexF64[small 0; 1+im 1-im]
        u, v = aca(i -> a[i, :], j -> a[:, j], 2, 2, 1e-6)
        state = ACAState([1, 2], [1, 2], false)
        aca_batched([state], (ii, jj, _) -> [a[i,j] for (i,j) in zip(ii,jj)], 1e-6)
        for product in (u*v', reduce(hcat,state.U)*reduce(hcat,state.V)')
            e = errors(product, a)
            @test e.relative <= 1e-12 && iszero(e.absolute)
        end
    end
end

include("coupled.jl")
include("dumps.jl")
@testset "Independent H-matrix CPU reference" begin
    @testset "Error metric" begin
        e = errors([1im, -1e-12, 2e-8], [1im, 1e-12, 0])
        @test e == (relative=2., absolute=2e-8, zeros=1)
    end
    @testset "Scalar and batched ACA" test_aca()
    @testset "Near, far and reflected operators" test_blocks_and_krylov()
    @testset "Coupled outputs" test_coupled()
    @testset "Reference dump identity" test_dump_contract()
end
