# Roadmap step 6, part 2: the H-matrix BEM solver (hmatrix_dev/hmat_engine_solver.jl) plugged into the
# engine hook `build_condensed_coupled_system(...; bem_solver=...)`, against the dense Float64 condensed
# path on the engine's test fixture (femvolume + exterior_conforming), for every output and excitation.
# Run from src/beat_engine/julia_local with FMM3D, Krylov, LinearMaps, KernelAbstractions, Adapt on
# JULIA_LOAD_PATH:
#   julia --project=. hmatrix_dev/tests/hmat_engine_hook_test.jl
# This is a contract and accuracy check on a small fixture, not an h343 qualification and not a
# performance measurement.
const JL_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(JL_ROOT, "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(JL_ROOT, "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(JL_ROOT, "tests", "coupled_condensed_test_setup.jl"))
include(joinpath(JL_ROOT, "src", "BeatEngineFmm.jl")); using .BeatEngineFmm
include(joinpath(JL_ROOT, "hmatrix_dev", "hmat_engine_solver.jl"))
using Test

# Pointwise complex relative error over nonzero reference entries; exact reference zeros separately
# as absolute error (same metric as tests/external_bem_solver_tests.jl, REVIEW-23-51d94c1).
function hook_pointwise_error(reference, candidate)
    r = ComplexF64.(vec(reference)); c = ComplexF64.(vec(candidate))
    length(r) == length(c) || error("Output lengths differ: $(length(r)) vs $(length(c)).")
    nz = .!iszero.(r)
    (relative=maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz]); init=0.0),
     normwise=norm(c .- r) / max(norm(r), floatmin(Float64)),
     absolute_at_zero=maximum(abs.(c[.!nz]); init=0.0), zero_count=count(!, nz), scale=maximum(abs, r; init=0.0))
end

const HOOK_OUTPUTS = (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux, :diaphragm_velocity, :voice_coil_current)
const HOOK_TOL = parse(Float64, get(ENV, "HMAT_TOL", "1e-6"))
# Acceptance bound for this fixture at HMAT_TOL=1e-6 and GMRES rtol 1e-9 (pointwise, every output).
const HOOK_BOUND = parse(Float64, get(ENV, "HMAT_HOOK_BOUND", "1e-4"))

with_baseline_cpu_coupled() do
    fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    imap = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    rtag = physical_tag(fem, 2, "Radiator")
    tr = ElectrodynamicTransducer{Float64}("component:test", [rtag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
        2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0)
    build(f; extra...) = build_condensed_coupled_system(fem, bem, imap, f, 343.0, 1.21;
        quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER, bem_backend=:cpu, extra...)
    velocity = (kind=:normal_velocity, radiator_tag=rtag, amplitude=ComplexF64(1, 0))
    voltage = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))
    voltage_b = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(0.3, -0.7))
    cases = (
        (name="no transducer 500 Hz", f=500.0, transducers=ElectrodynamicTransducer{Float64}[], excitations=[velocity]),
        (name="transducer 700 Hz", f=700.0, transducers=[tr], excitations=[voltage, voltage_b]),
        (name="transducer 2000 Hz", f=2000.0, transducers=[tr], excitations=[voltage, voltage_b]),
    )
    @testset "H-matrix BEM solver in the engine hook (tol $(HOOK_TOL))" begin
        for case in cases
            @testset "$(case.name)" begin
                solves = HMatCoupledSolve[]
                dense = build(case.f; transducers=case.transducers)
                captured = Ref{Any}(nothing)
                hook = hmatrix_bem_solver(; tol=HOOK_TOL, rtol=1e-9, solves=solves)
                hsys = build(case.f; transducers=case.transducers,
                    bem_solver=ctx -> (h = hook(ctx); captured[] = (context=ctx, hook=h); h))
                try
                    @test hsys.bem_solver == :external
                    @test hsys.coupled_bem_assembly == :external
                    @test length(solves) == 1
                    S = only(solves)
                    @printf("  %s: n %d, H(A) storage %.3f of dense, H(C) %d columns %.3f of dense\n",
                        case.name, size(S, 1), S.build.storage_A, S.build.flux_columns, S.build.storage_C)
                    reference = solve_condensed_coupled_excitations(dense, case.excitations)
                    candidate = solve_condensed_coupled_excitations(hsys, case.excitations)
                    @test !isempty(S.stats) && all(s -> s.solved, S.stats)
                    for s in S.stats
                        @printf("    GMRES %d iterations, scaled relative residual %.2e\n", s.iterations, s.residual)
                    end
                    @test length(reference) == length(candidate) == length(case.excitations)
                    # flux_operator_mul (prescribed BEM normal data) against the dense operator C = -R.
                    ctx = captured[].context
                    ops = BeatEngineCoupledCondensed._assemble_condensed_bem_operators(
                        ctx.bem_mesh, ctx.prepared, ctx.wavenumber, ctx.singular_order)
                    _, R, _ = BeatEngineCoupledCondensed._combine_condensed_bem_operators!(ops, ctx.prepared, ctx.wavenumber)
                    ndp0 = size(R, 2)
                    Qtest = zeros(ComplexF64, ndp0, 3)
                    Qtest[1:3:end, 1] .= 1.0; Qtest[:, 2] .= range(-1, 1; length=ndp0) .* (0.4 - 0.9im)
                    Cq = captured[].hook.flux_operator_mul(Qtest); Cd = -Matrix(R) * Qtest
                    eC = norm(Cq .- Cd) / norm(Cd)
                    @printf("    flux_operator_mul vs dense C: normwise %.2e, zero column %s\n", eC, iszero(Cq[:, 3]))
                    @test eC < 1e-10
                    @test iszero(Cq[:, 3])
                    for (column, (r, c)) in enumerate(zip(reference, candidate))
                        checked = 0
                        for name in HOOK_OUTPUTS
                            hasproperty(r, name) && hasproperty(c, name) || continue
                            rv = getproperty(r, name); rv isa AbstractArray{<:Number} || continue
                            e = hook_pointwise_error(rv, getproperty(c, name)); checked += 1
                            @printf("    excitation %d %-19s pointwise %.2e  normwise %.2e%s\n", column, name,
                                e.relative, e.normwise,
                                e.zero_count > 0 ? @sprintf("  (%d exact zeros, max abs %.2e)", e.zero_count, e.absolute_at_zero) : "")
                            @test e.relative < HOOK_BOUND
                            @test e.absolute_at_zero <= HOOK_BOUND * max(e.scale, floatmin(Float64))
                        end
                        @test checked >= 4
                    end
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(hsys)
                end
            end
        end
        # REVIEW-23-5ea59ac: actual solves of prescribed BEM sources (with the driver in the system),
        # matrix right-hand sides with a zero column, and a forced GMRES failure.
        @testset "prescribed BEM sources, matrix RHS, forced non-convergence (700 Hz)" begin
            faces = length(bem.faces)
            prescribed = sparse([1, 2, faces], [1, 1, 2], [1.0, -0.5, 2.0], faces, 2)
            sources = [(kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float64[],
                        bem_source_index=1, amplitude=ComplexF64(1, 0)),
                       (kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float64[],
                        bem_source_index=2, amplitude=ComplexF64(-0.4, 0.9))]
            solves = HMatCoupledSolve[]
            dense = build(700.0; transducers=[tr], prescribed_bem_normal_velocity=prescribed)
            hsys = build(700.0; transducers=[tr], prescribed_bem_normal_velocity=prescribed,
                bem_solver=hmatrix_bem_solver(; tol=HOOK_TOL, rtol=1e-9, solves=solves))
            failing = HMatCoupledSolve[]
            hfail = build(700.0; transducers=[tr],
                bem_solver=hmatrix_bem_solver(; tol=HOOK_TOL, rtol=1e-9, maxiter=1, solves=failing))
            try
                reference = solve_condensed_coupled_excitations(dense, sources)
                candidate = solve_condensed_coupled_excitations(hsys, sources)
                @test all(s -> s.solved, only(solves).stats)
                for (column, (r, c)) in enumerate(zip(reference, candidate))
                    checked = 0
                    for name in HOOK_OUTPUTS
                        hasproperty(r, name) && hasproperty(c, name) || continue
                        rv = getproperty(r, name); rv isa AbstractArray{<:Number} || continue
                        e = hook_pointwise_error(rv, getproperty(c, name)); checked += 1
                        @printf("    BEM source %d %-19s pointwise %.2e  normwise %.2e%s\n", column, name,
                            e.relative, e.normwise,
                            e.zero_count > 0 ? @sprintf("  (%d exact zeros, max abs %.2e)", e.zero_count, e.absolute_at_zero) : "")
                        @test e.relative < HOOK_BOUND
                        @test e.absolute_at_zero <= HOOK_BOUND * max(e.scale, floatmin(Float64))
                    end
                    @test checked >= 6
                end
                # Matrix RHS: zero column stays exactly zero, the others reach the scaled residual.
                S = only(solves); n = size(S, 1); D = Diagonal(S.base.row_scale)
                r = ComplexF64.(sin.(1:n) .+ im .* cos.(0.7 .* (1:n)))
                R = hcat(r, zeros(ComplexF64, n), (0.3 - 0.7im) .* reverse(r))
                X = S \ R
                @test size(X) == size(R) && iszero(X[:, 2])
                for c in (1, 3)
                    res = norm(D * (_hmat_coupled_mul(S, X[:, c]) .- R[:, c])) / norm(D * R[:, c])
                    @printf("    matrix RHS column %d: scaled residual %.2e\n", c, res)
                    @test res <= 1e-8
                end
                @test vec(S \ r) ≈ X[:, 1] rtol = 1e-7
                # Forced failure: unconverged columns are reported, never returned as a solution.
                F = only(failing)
                err = try
                    F \ R; nothing
                catch e
                    e
                end
                @test err isa FmmNotConvergedError
                @test err.columns == [1, 3]
                @test length(err.residuals) == 2 && all(>(1e-9), err.residuals)
                @test size(err.iterates) == size(R) && iszero(err.iterates[:, 2])
                @test_throws FmmNotConvergedError solve_condensed_coupled_excitations(hfail, [voltage])
            finally
                release_condensed_coupled_system!(dense)
                release_condensed_coupled_system!(hsys)
                release_condensed_coupled_system!(hfail)
            end
        end
        @testset "Float32 systems are rejected" begin
            hook = hmatrix_bem_solver()
            @test_throws ErrorException hook((bem_mesh=bem, prepared=nothing, wavenumber=1.0, singular_order=1,
                symmetry_mode=:off, scalar_type=Float32))
        end
    end
end
