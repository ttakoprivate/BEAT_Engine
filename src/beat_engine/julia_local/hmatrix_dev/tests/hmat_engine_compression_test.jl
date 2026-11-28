# Roadmap step 6: H-matrix BEM solver in the engine hook on a BEM mesh that actually compresses.
# On the engine fixture alone (femvolume + exterior_conforming, 1214 BEM vertices) H(A) keeps ~98 % of
# the dense storage, so hmat_engine_hook_test.jl and hmat_driver_switch_test.jl check wiring and the
# near field, not low-rank blocks (REVIEW-23-a696772, "Grenzen"). Here a passive rigid icosphere
# scatterer is added to the BEM mesh on its own free physical tag (not the interface tag and not a
# transducer BEM tag, so homogeneous Neumann: no prescribed motion, no force coupling; REVIEW-23-378cba5
# point 1), well separated from the enclosure. The enclosure<->sphere far blocks of H(A) become ACA
# low-rank blocks (at subdiv 3 / 700 Hz the sphere is too small for sphere<->sphere far blocks; those
# are not claimed). Every output of the driven system (both voltage excitations) is compared pointwise
# (complex relative error) against the dense Float64 condensed path on the same extended mesh, and the
# H(A) / H(C) storage is required to be clearly below dense.
#
# Run from src/beat_engine/julia_local with FMM3D, Krylov, LinearMaps, KernelAbstractions, Adapt on
# JULIA_LOAD_PATH:
#   julia --project=. hmatrix_dev/tests/hmat_engine_compression_test.jl
# Optional: HMAT_COMPRESSION_SUBDIV (icosphere subdivisions, default 3), HMAT_TOL (default 1e-6).
# An accuracy check of compressed blocks on CPU, not an h343 qualification and not a performance or
# memory measurement (storage fractions are block counts of this mesh, not process memory).
const JL_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
include(joinpath(JL_ROOT, "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(JL_ROOT, "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
include(joinpath(JL_ROOT, "tests", "coupled_condensed_test_setup.jl"))
include(joinpath(JL_ROOT, "src", "BeatEngineFmm.jl")); using .BeatEngineFmm
include(joinpath(JL_ROOT, "hmatrix_dev", "hmat_engine_solver.jl"))
include(joinpath(JL_ROOT, "tests", "fmm_test_meshes.jl"))
using Test, Printf

const COMPRESSION_TOL = parse(Float64, get(ENV, "HMAT_TOL", "1e-6"))
const COMPRESSION_SUBDIV = parse(Int, get(ENV, "HMAT_COMPRESSION_SUBDIV", "3"))
# Same pointwise bound as hmat_engine_hook_test.jl at HMAT_TOL=1e-6 and GMRES rtol 1e-9.
const COMPRESSION_BOUND = parse(Float64, get(ENV, "HMAT_HOOK_BOUND", "1e-4"))
# The test is only meaningful if low-rank blocks carry a real part of the operator.
const COMPRESSION_MAX_STORAGE_A = 0.8

const COMPRESSION_OUTPUTS = (:fem_pressure, :bem_pressure, :bem_neumann, :interface_flux,
                             :diaphragm_velocity, :voice_coil_current)

# Pointwise complex relative error over nonzero reference entries; exact reference zeros separately as
# absolute error (metric of hmat_engine_hook_test.jl).
function compression_pointwise_error(reference, candidate)
    r = ComplexF64.(vec(reference)); c = ComplexF64.(vec(candidate))
    length(r) == length(c) || error("Output lengths differ: $(length(r)) vs $(length(c)).")
    nz = .!iszero.(r)
    (relative=maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz]); init=0.0),
     normwise=norm(c .- r) / max(norm(r), floatmin(Float64)),
     absolute_at_zero=maximum(abs.(c[.!nz]); init=0.0), zero_count=count(!, nz), scale=maximum(abs, r; init=0.0))
end

"""Structural problems of a result pair: column counts and the six required numeric outputs per column."""
function compression_output_problems(reference, candidate, n_excitations)
    problems = String[]
    length(reference) == n_excitations || push!(problems, "reference has $(length(reference)) columns, expected $(n_excitations)")
    length(candidate) == n_excitations || push!(problems, "candidate has $(length(candidate)) columns, expected $(n_excitations)")
    for (label, results) in (("reference", reference), ("candidate", candidate)), column in eachindex(results)
        for name in COMPRESSION_OUTPUTS
            ok = hasproperty(results[column], name) && getproperty(results[column], name) isa AbstractArray{<:Number}
            ok || push!(problems, "$(label) column $(column) lacks numeric $(name)")
        end
    end
    problems
end

@testset "output guard rejects missing columns and outputs" begin
    full = NamedTuple{COMPRESSION_OUTPUTS}(ntuple(_ -> ComplexF64[1], length(COMPRESSION_OUTPUTS)))
    partial = (fem_pressure=ComplexF64[1], bem_pressure=ComplexF64[1], bem_neumann=ComplexF64[1], interface_flux=ComplexF64[1])
    @test isempty(compression_output_problems([full, full], [full, full], 2))
    # Counterexample of REVIEW-23-378cba5 point 3: one candidate column without the transducer outputs.
    @test length(compression_output_problems([full, full], [partial], 2)) == 3
    @test length(compression_output_problems([full, full], [full, partial], 2)) == 2
end

"""Append a closed rigid icosphere (outward normals, `tag`) to a BEM mesh; returns the new mesh."""
function append_rigid_sphere(bem::BoundaryMesh{Float64}, radius, subdivisions, centre, tag)
    s = _fmm_icosphere(radius, subdivisions; centre=centre)
    offset = length(bem.vertices)
    return BoundaryMesh(vcat(bem.vertices, s.vertices),
                        vcat(bem.faces, [(a + offset, b + offset, c + offset) for (a, b, c) in s.faces]),
                        vcat(bem.physical_tags, fill(tag, length(s.faces))))
end

@testset "sphere helper: closed, outward, separate" begin
    s = _fmm_icosphere(0.08, 2; centre=SVector(0.0, 0.0, 0.5))
    # Outward normals: the enclosed volume (divergence theorem) is positive and close to 4/3 pi r^3.
    vol = sum(dot(s.centroids[i], s.normals[i]) * s.areas[i] for i in eachindex(s.faces)) / 3
    @test 0.95 * (4π / 3) * 0.08^3 < vol < (4π / 3) * 0.08^3
end

with_baseline_cpu_coupled() do
    fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    base = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    interface_tag = 2
    rtag = physical_tag(fem, 2, "Radiator")
    tr = ElectrodynamicTransducer{Float64}("component:test", [rtag], [1.0], [1], [-1.0], SVector(0.0, 0.0, 1.0),
        2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0)
    # The sphere gets a free tag: not the interface and not a driven (transducer) BEM tag.
    sphere_tag = maximum(base.physical_tags) + 1
    @test !(sphere_tag in base.physical_tags) && sphere_tag != interface_tag && !(sphere_tag in tr.bem_boundary_tags)
    # Enclosure spans about 0.25 x 0.15 x 0.12 m (z from -0.025 to 0.095); the sphere sits above it.
    centre = SVector(0.0, 0.0, 0.40)
    bem = append_rigid_sphere(base, 0.10, COMPRESSION_SUBDIV, centre, sphere_tag)
    nB0 = length(base.vertices); nF0 = length(base.faces)
    sphere_vertices = nB0+1:length(bem.vertices); sphere_faces = nF0+1:length(bem.faces)
    gap = minimum(norm(v - centre) for v in base.vertices) - 0.10
    @printf("  BEM mesh: %d fixture + %d sphere vertices, gap enclosure-sphere %.3f m\n",
            length(base.vertices), length(bem.vertices) - length(base.vertices), gap)
    @test gap > 0.2
    imap = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), interface_tag)
    @testset "sphere is passive (homogeneous Neumann)" begin
        @test isempty(intersect(imap.bem_face_indices, sphere_faces))
        ops = assemble_transducer_operators(fem, bem, [tr])
        @test iszero(ops.bem_normal_velocity[sphere_faces, :])
        @test iszero(ops.bem_surface[sphere_vertices, :])
        @test iszero(ops.bem_force[sphere_vertices, :])
        @test !iszero(ops.bem_normal_velocity[1:nF0, :])   # the driver still moves the fixture faces
    end
    build(f; extra...) = build_condensed_coupled_system(fem, bem, imap, f, 343.0, 1.21;
        quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER, bem_backend=:cpu,
        transducers=[tr], extra...)
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0)),
                   (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(0.3, -0.7))]
    worst = Dict{Tuple{Symbol,Int},Float64}()
    @testset "compressed H-matrix vs dense Float64 (tol $(COMPRESSION_TOL), subdiv $(COMPRESSION_SUBDIV))" begin
        for f in (700.0, 2000.0)
            @testset "$(f) Hz" begin
                solves = HMatCoupledSolve[]
                dense = build(f)
                hsys = build(f; bem_solver=hmatrix_bem_solver(; tol=COMPRESSION_TOL, rtol=1e-9, solves=solves))
                try
                    @test hsys.bem_solver == :external
                    S = only(solves)
                    @printf("  %.0f Hz: n %d, H(A) storage %.3f of dense, H(C) %d columns %.3f of dense\n",
                            f, size(S, 1), S.build.storage_A, S.build.flux_columns, S.build.storage_C)
                    @test S.build.storage_A < COMPRESSION_MAX_STORAGE_A
                    @test S.build.storage_C < 1.0
                    # Which couplings are compressed: low-rank blocks of H(A) by enclosure / sphere rows and columns.
                    on_sphere(ix) = all(>(nB0), ix); on_enclosure(ix) = all(<=(nB0), ix)
                    cross = count(b -> (on_enclosure(b.rows) && on_sphere(b.cols)) || (on_sphere(b.rows) && on_enclosure(b.cols)), S.HA.lr)
                    within = count(b -> on_sphere(b.rows) && on_sphere(b.cols), S.HA.lr)
                    @printf("    H(A) low-rank blocks: %d enclosure<->sphere, %d sphere<->sphere, %d total\n",
                            cross, within, length(S.HA.lr))
                    @test cross > 0
                    reference = solve_condensed_coupled_excitations(dense, excitations)
                    candidate = solve_condensed_coupled_excitations(hsys, excitations)
                    @test all(s -> s.solved, S.stats)
                    for s in S.stats
                        @printf("    GMRES %d iterations, scaled relative residual %.2e\n", s.iterations, s.residual)
                    end
                    # The sphere must matter: its surface pressure is not negligible against the enclosure's.
                    p = ComplexF64.(reference[1].bem_pressure)
                    ratio = maximum(abs, p[nB0+1:end]) / maximum(abs, p[1:nB0])
                    @printf("    max |p| sphere / enclosure %.3e\n", ratio)
                    @test ratio > 1e-3
                    # Both excitation columns, every required output (REVIEW-23-378cba5 point 3).
                    problems = compression_output_problems(reference, candidate, length(excitations))
                    foreach(println, problems)
                    @test isempty(problems)
                    for column in (isempty(problems) ? eachindex(excitations) : (1:0))
                        r = reference[column]; c = candidate[column]
                        # Homogeneous Neumann on the passive sphere, in both paths.
                        @test all(iszero, r.bem_neumann[sphere_faces])
                        @test all(iszero, c.bem_neumann[sphere_faces])
                        for name in COMPRESSION_OUTPUTS
                            e = compression_pointwise_error(getproperty(r, name), getproperty(c, name))
                            worst[(name, column)] = max(get(worst, (name, column), 0.0), e.relative)
                            @printf("    excitation %d %-19s pointwise %.2e  normwise %.2e%s\n", column, name,
                                    e.relative, e.normwise,
                                    e.zero_count > 0 ? @sprintf("  (%d exact zeros, max abs %.2e)", e.zero_count, e.absolute_at_zero) : "")
                            @test e.relative < COMPRESSION_BOUND
                            @test e.absolute_at_zero <= COMPRESSION_BOUND * max(e.scale, floatmin(Float64))
                        end
                    end
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(hsys)
                end
            end
        end
    end
    println("  max pointwise complex relative error vs dense Float64 over both frequencies:")
    for name in COMPRESSION_OUTPUTS, column in 1:2
        haskey(worst, (name, column)) && @printf("    %-19s excitation %d: %.3e\n", name, column, worst[(name, column)])
    end
end
