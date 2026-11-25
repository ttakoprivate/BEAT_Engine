# Roadmap step 6, first part: the engine hook `bem_solver` of `build_condensed_coupled_system`.
# A test solver that rebuilds the dense BEM rows from the hook's `rest` matrix and factors them
# must reproduce the dense condensed path (Float64) to rounding, for every output and excitation.
# This checks the hook's layout and sign contract; the H-matrix solver plugs into the same hook.
include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
using Test

const EXT_ROOT = CONDENSED_FIXTURE_ROOT

# Dense reference implementation of the hook contract (see build_condensed_coupled_system).
function dense_hook_solver(calls::Ref{Int}, rests::Vector{Any})
    return context -> begin
        calls[] += 1
        operators = BeatEngineCoupledCondensed._assemble_condensed_bem_operators(
            context.bem_mesh, context.prepared, context.wavenumber, context.singular_order)
        A, R, _ = BeatEngineCoupledCondensed._combine_condensed_bem_operators!(operators, context.prepared, context.wavenumber)
        # R is the dense path's right-hand operator -C.
        C = -Matrix(R)
        (
            flux_operator_mul=Q -> C * Q,
            factorize=(rest, layout) -> begin
                push!(rests, (rest=rest, layout=layout))
                M = Matrix(rest)
                B = layout.bem_range
                M[B, B] .+= A
                M[B, layout.flux_range] .+= C * layout.bem_flux
                isempty(layout.mechanical_range) || (M[B, layout.mechanical_range] .+= C * layout.bem_motion_flux)
                lu(M)
            end,
        )
    end
end

# Acceptance metric (roadmap): pointwise complex relative error |c - r| / |r| over the nonzero
# reference entries; exact reference zeros are reported separately as an absolute error in the
# output's unit. A normalized maximum would hide a large error at a small entry (REVIEW-23-51d94c1).
function pointwise_complex_error(reference, candidate)
    r = ComplexF64.(vec(reference)); c = ComplexF64.(vec(candidate))
    length(r) == length(c) || error("Output lengths differ: $(length(r)) vs $(length(c)).")
    nonzero = .!iszero.(r)
    relative = maximum(abs.(c[nonzero] .- r[nonzero]) ./ abs.(r[nonzero]); init=0.0)
    absolute_at_zero = maximum(abs.(c[.!nonzero]); init=0.0)
    return (relative=relative, absolute_at_zero=absolute_at_zero, zero_count=count(!, nonzero),
            scale=maximum(abs, r; init=0.0))
end

const OUTPUT_UNITS = Dict(:fem_pressure => "Pa", :bem_pressure => "Pa", :bem_neumann => "Pa/m",
    :interface_flux => "Pa/m", :diaphragm_velocity => "m/s", :voice_coil_current => "A")

function compare_solutions(reference, candidate)
    worst = Dict{Symbol,Any}()
    for name in propertynames(reference)
        value = getproperty(reference, name)
        value isa AbstractArray{<:Number} || continue
        eltype(value) <: Complex || eltype(value) <: AbstractFloat || continue
        hasproperty(candidate, name) || continue
        worst[name] = pointwise_complex_error(value, getproperty(candidate, name))
    end
    return worst
end

# Every output and excitation: pointwise relative error below `tolerance`, and at exact reference
# zeros an absolute error below `tolerance` times the output's largest reference magnitude.
function check_solutions(label, references, candidates; tolerance=1e-10)
    @test length(references) == length(candidates)
    for (column, (r, c)) in enumerate(zip(references, candidates))
        worst = compare_solutions(r, c)
        @test !isempty(worst)
        for (output, e) in sort(collect(worst); by=first)
            unit = get(OUTPUT_UNITS, output, "")
            println("  $label excitation $column $output: max pointwise complex relative error $(e.relative)",
                    e.zero_count > 0 ? "; $(e.zero_count) exact reference zeros, max absolute error $(e.absolute_at_zero) $unit" : "")
            @test e.relative < tolerance
            @test e.absolute_at_zero <= tolerance * max(e.scale, floatmin(Float64))
        end
    end
end

# Shifts a mesh as a whole (normals unchanged) so a symmetry fixture lies in the positive half-space.
shifted_boundary_mesh(m::BoundaryMesh, d) = BoundaryMesh(m.vertices .+ Ref(d), m.faces, m.physical_tags,
    m.centroids .+ Ref(d), m.normals, m.areas, [map(v -> v + d, f) for f in m.face_vertices])
shifted_volume_mesh(m, d) = typeof(m)(m.vertices .+ Ref(d), m.tetrahedra, m.tetra_physical_tags, m.boundary_faces,
    m.boundary_physical_tags, m.physical_names, m.quadratic_tetrahedra, m.quadratic_boundary_faces)

@testset "pointwise metric (REVIEW-23-51d94c1)" begin
    # A large error at a small entry is not hidden by the largest entry.
    @test pointwise_complex_error(ComplexF64[1, 1e-12], ComplexF64[1, 2e-12]).relative ≈ 1.0
    # Phase reversal: 0 dB level difference, complex error 2.
    @test pointwise_complex_error(ComplexF64[1, -1], ComplexF64[-1, 1]).relative ≈ 2.0
    # Mixed zero / nonzero reference: zeros go to the absolute error, not into a division.
    e = pointwise_complex_error(ComplexF64[0, 2, 0], ComplexF64[1e-3, 2, -2e-3im])
    @test e.relative == 0.0 && e.absolute_at_zero ≈ 2e-3 && e.zero_count == 2
    @test pointwise_complex_error(zeros(ComplexF64, 3), zeros(ComplexF64, 3)) ==
          (relative=0.0, absolute_at_zero=0.0, zero_count=3, scale=0.0)
end

with_baseline_cpu_coupled() do
    @testset "External BEM solver hook (roadmap step 6)" begin
        fem_mesh = load_gmsh41_volume(joinpath(EXT_ROOT, "femvolume.msh"), 0.001)
        bem_mesh = load_gmsh22_with_tags(joinpath(EXT_ROOT, "exterior_conforming.msh"), 0.001)
        interface_map = build_conforming_interface_map(fem_mesh, bem_mesh, physical_tag(fem_mesh, 2, "Interface"), 2)
        radiator_tag = physical_tag(fem_mesh, 2, "Radiator")
        transducer = ElectrodynamicTransducer{Float64}("component:test", [radiator_tag], [1.0], [1], [-1.0],
            SVector(0.0, 0.0, 1.0), 2.0, 1, 6.0, 0.0005, 7.0, 0.015, 0.0005, 1.0)
        build(frequency; extra...) = build_condensed_coupled_system(fem_mesh, bem_mesh, interface_map, frequency, 343.0, 1.21;
            quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER, bem_backend=:cpu, extra...)
        velocity = (kind=:normal_velocity, radiator_tag=radiator_tag, amplitude=ComplexF64(1, 0))
        voltage = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(1, 0))
        voltage_b = (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=ComplexF64(0.3, -0.7))

        cases = (
            (name="no transducer 500 Hz", f=500.0, transducers=ElectrodynamicTransducer{Float64}[], excitations=[velocity]),
            (name="transducer 700 Hz", f=700.0, transducers=[transducer], excitations=[voltage, voltage_b]),
            (name="transducer 2000 Hz", f=2000.0, transducers=[transducer], excitations=[voltage, voltage_b]),
        )
        for case in cases
            @testset "$(case.name)" begin
                calls = Ref(0); rests = Any[]
                dense = build(case.f; transducers=case.transducers)
                external = build(case.f; transducers=case.transducers, bem_solver=dense_hook_solver(calls, rests))
                try
                    @test dense.bem_solver == :dense
                    @test external.bem_solver == :external
                    @test dense.coupled_bem_assembly == :operators
                    @test external.coupled_bem_assembly == :external
                    @test calls[] == 1
                    @test length(rests) == 1
                    rest = only(rests).rest; layout = only(rests).layout
                    @test rest isa SparseMatrixCSC
                    # No A or C entry reaches the system the hook sees.
                    @test nnz(rest[layout.bem_range, :]) == 0
                    @test layout.bem_range == dense.bem_range && layout.flux_range == dense.flux_range
                    @test external.interface_elimination == :none
                    reference = solve_condensed_coupled_excitations(dense, case.excitations)
                    candidate = solve_condensed_coupled_excitations(external, case.excitations)
                    check_solutions(case.name, reference, candidate)
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(external)
                end
            end
        end

        @testset "requested flux elimination is forced off" begin
            withenv("BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1", "BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1") do
                calls = Ref(0); rests = Any[]
                dense = build(700.0; transducers=[transducer])
                external = build(700.0; transducers=[transducer], bem_solver=dense_hook_solver(calls, rests))
                try
                    @test external.interface_elimination == :none
                    @test any(occursin("external BEM solver", reason) for reason in external.optimization_fallback_reasons)
                    check_solutions("flux elimination requested",
                        solve_condensed_coupled_excitations(dense, [voltage, voltage_b]),
                        solve_condensed_coupled_excitations(external, [voltage, voltage_b]); tolerance=1e-9)
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(external)
                end
            end
        end

        @testset "prescribed BEM normal velocity right-hand side" begin
            faces = length(bem_mesh.faces)
            prescribed = sparse([1, 2, faces], [1, 1, 2], [1.0, -0.5, 2.0], faces, 2)
            dense = build(700.0; prescribed_bem_normal_velocity=prescribed)
            external = build(700.0; prescribed_bem_normal_velocity=prescribed, bem_solver=dense_hook_solver(Ref(0), Any[]))
            try
                @test size(external.prescribed_bem_rhs) == size(dense.prescribed_bem_rhs) == (length(bem_mesh.vertices), 2)
                e = pointwise_complex_error(dense.prescribed_bem_rhs, external.prescribed_bem_rhs)
                @test e.relative < 1e-12 && e.absolute_at_zero <= 1e-12 * e.scale
                # Actual solves of prescribed BEM excitations, not only the right-hand side.
                sources = [(kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float64[],
                            bem_source_index=1, amplitude=ComplexF64(1, 0)),
                           (kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float64[],
                            bem_source_index=2, amplitude=ComplexF64(-0.4, 0.9))]
                check_solutions("prescribed BEM velocity", solve_condensed_coupled_excitations(dense, sources),
                    solve_condensed_coupled_excitations(external, sources))
            finally
                release_condensed_coupled_system!(dense)
                release_condensed_coupled_system!(external)
            end
        end

        # Symmetry, wall impedance and thermoviscous walls (cases from REVIEW-23-51d94c1). The coupled
        # fixture is shifted as a whole into the positive half-space: an algebraic contract fixture.
        shift = SVector(0.3, 0.3, 0.0)
        fem_s = shifted_volume_mesh(fem_mesh, shift)
        bem_s = shifted_boundary_mesh(bem_mesh, shift)
        map_s = build_conforming_interface_map(fem_s, bem_s, physical_tag(fem_s, 2, "Interface"), 2)
        wall_tag = physical_tag(fem_s, 2, "Volume_boundary")
        build_s(; extra...) = build_condensed_coupled_system(fem_s, bem_s, map_s, 700.0, 343.0, 1.21;
            quadrature_order=CONDENSED_QUADRATURE_ORDER, singular_order=CONDENSED_SINGULAR_ORDER, bem_backend=:cpu,
            transducers=[transducer], extra...)
        for (name, extra) in (
            ("symmetry x", (symmetry_mode=:x,)),
            ("symmetry xy", (symmetry_mode=:xy,)),
            ("wall impedance", (wall_impedances=[(tag=wall_tag, thickness_m=0.02, flow_resistivity_pa_s_per_m2=12000.0)],)),
            ("thermoviscous wall", (thermoviscous_walls=[(tag=wall_tag, boundary_id="boundary:wall")],)),
        )
            @testset "$name" begin
                dense = build_s(; extra...)
                external = build_s(; extra..., bem_solver=dense_hook_solver(Ref(0), Any[]))
                try
                    @test external.bem_solver == :external
                    check_solutions(name, solve_condensed_coupled_excitations(dense, [voltage, voltage_b]),
                        solve_condensed_coupled_excitations(external, [voltage, voltage_b]))
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(external)
                end
            end
        end

        @testset "unsupported combinations are rejected" begin
            solver = dense_hook_solver(Ref(0), Any[])
            @test_throws ErrorException build(500.0; bem_solver=solver, retain_interface_radiation=true)
            @test_throws ErrorException build(500.0; bem_solver=solver, bem_operators=() -> nothing)
        end
    end
end
