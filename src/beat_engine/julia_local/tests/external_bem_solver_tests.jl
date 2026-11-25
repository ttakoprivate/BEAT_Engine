# Roadmap step 6, first part: the engine hook `bem_solver` of `build_condensed_coupled_system`.
# A test solver that rebuilds the dense BEM rows from the hook's `rest` matrix and factors them
# must reproduce the dense condensed path (Float64) to rounding, for every output and excitation.
# This checks the hook's layout and sign contract; the H-matrix solver plugs into the same hook.
include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
isdefined(@__MODULE__, :BeatEngineInterfaceRadiation) ||
    include(joinpath(@__DIR__, "..", "src", "BeatEngineInterfaceRadiation.jl"))
using .BeatEngineInterfaceRadiation
using Test

const EXT_ROOT = CONDENSED_FIXTURE_ROOT

# Dense reference implementation of the hook contract (see build_condensed_coupled_system).
function dense_hook_solver(calls::Ref{Int}, rests::Vector{Any}; exterior_solve::Bool=true)
    return context -> begin
        calls[] += 1
        operators = BeatEngineCoupledCondensed._assemble_condensed_bem_operators(
            context.bem_mesh, context.prepared, context.wavenumber, context.singular_order)
        A, R, _ = BeatEngineCoupledCondensed._combine_condensed_bem_operators!(operators, context.prepared, context.wavenumber)
        # R is the dense path's right-hand operator -C.
        C = -Matrix(R)
        exterior = exterior_solve ? (; exterior_solve=(let F = lu(A); Rhs -> F \ Rhs; end)) : (;)
        (;
            exterior...,
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

# Remainder column of `interface_radiation_traces` (total output minus the interface traces). It is a
# subtraction, so some reference entries are only its rounding residue. Report both scales
# (REVIEW-23-a2b222e): the pointwise complex relative error over all nonzero entries and the maximum
# absolute error in the output's unit. Acceptance splits the entries at `cancellation * total_scale`:
# entries at or above it are checked pointwise; entries below it (rounding residue and exact zeros) get
# a separate absolute bound against the total output's largest magnitude. Neither replaces the other.
function remainder_complex_error(reference, candidate, total_scale; cancellation=1e-12)
    r = ComplexF64.(vec(reference)); c = ComplexF64.(vec(candidate))
    length(r) == length(c) || error("Output lengths differ: $(length(r)) vs $(length(c)).")
    err = abs.(c .- r)
    nonzero = .!iszero.(r)
    significant = abs.(r) .>= cancellation * total_scale
    return (relative=maximum(err[nonzero] ./ abs.(r[nonzero]); init=0.0),
            relative_significant=maximum(err[significant] ./ abs.(r[significant]); init=0.0),
            significant_count=count(significant), residue_count=count(nonzero .& .!significant),
            zero_count=count(!, nonzero), absolute=maximum(err; init=0.0),
            absolute_at_zero=maximum(err[.!nonzero]; init=0.0),
            absolute_residue=maximum(err[.!significant]; init=0.0), total_scale=total_scale)
end

function print_remainder_error(io, label, e, unit)
    println(io, "  ", label, ": pointwise all nonzero ", e.relative, ", pointwise significant (", e.significant_count,
            " entries >= 1e-12 max|total|) ", e.relative_significant, ", max abs ", e.absolute, " ", unit,
            "; residue entries ", e.residue_count, ", exact zeros ", e.zero_count, ", max abs on residue/zeros ",
            e.absolute_residue, " ", unit, " (at zeros ", e.absolute_at_zero, " ", unit, "), max |total| ",
            e.total_scale, " ", unit)
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

@testset "remainder metric (REVIEW-23-a2b222e)" begin
    # Counterexample from the review: a residue entry with a large pointwise error is reported, not hidden.
    reference = ComplexF64[2, 1e-14, 0, -1]
    candidate = ComplexF64[2 * (1 + 1e-9), 5e-14, 3e-14, -1]
    e = remainder_complex_error(reference, candidate, 2.0)
    @test isapprox(e.relative, 4.0; rtol=1e-6) && isapprox(e.relative_significant, 1e-9; rtol=1e-6)
    @test e.significant_count == 2 && e.residue_count == 1 && e.zero_count == 1
    @test isapprox(e.absolute, 2e-9; rtol=1e-6) && isapprox(e.absolute_residue, 4e-14; rtol=1e-6) && e.absolute_at_zero == 3e-14
    # A large relative error on a significant entry fails the pointwise part despite a tiny normalized error.
    bad = remainder_complex_error(ComplexF64[1, 1e-6], ComplexF64[1, 2e-6], 1.0)
    @test bad.relative_significant ≈ 1.0 && bad.absolute / bad.total_scale ≈ 1e-6
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

        # Interface radiation replay through the hook's `exterior_solve` and `flux_operator_mul`: the
        # traces of every interface source must equal the dense condensed replay (A \ (-C bem_flux q)).
        @testset "interface radiation replay" begin
            faces = length(bem_mesh.faces)
            prescribed = sparse([1, 2, faces], [1, 1, 1], [1.0, -0.5, 2.0], faces, 1)
            bem_source = (kind=:normal_velocity, fem_boundary_tags=Int[], fem_boundary_weights=Float64[],
                          bem_source_index=1, amplitude=ComplexF64(-0.4, 0.9))
            for (name, f, transducers, excitations) in (
                ("replay no transducer 500 Hz", 500.0, ElectrodynamicTransducer{Float64}[], [velocity, bem_source]),
                ("replay transducer 700 Hz", 700.0, [transducer], [voltage, voltage_b, bem_source]),
            )
                dense = build(f; transducers=transducers, prescribed_bem_normal_velocity=prescribed,
                              retain_interface_radiation=true)
                external = build(f; transducers=transducers, prescribed_bem_normal_velocity=prescribed,
                                 retain_interface_radiation=true, bem_solver=dense_hook_solver(Ref(0), Any[]))
                try
                    @test external.interface_radiation_replay.backend == :external
                    references = solve_condensed_coupled_excitations(dense, excitations)
                    candidates = solve_condensed_coupled_excitations(external, excitations)
                    for (column, (r, c)) in enumerate(zip(references, candidates))
                        n = length(r.interface_flux)
                        ranges = [1:2:n, 2:2:n]
                        for include_other in (false, true)
                            tr = interface_radiation_traces(dense, r, ranges; include_other=include_other)
                            tc = interface_radiation_traces(external, c, ranges; include_other=include_other)
                            @test size(tc.pressure) == size(tr.pressure)
                            for source in axes(tr.pressure, 2)
                                if include_other && source == size(tr.pressure, 2)
                                    # Remainder column: pointwise and absolute, see remainder_complex_error.
                                    ep = remainder_complex_error(tr.pressure[:, source], tc.pressure[:, source],
                                        maximum(abs, r.bem_pressure))
                                    en = remainder_complex_error(tr.normal_derivative[:, source], tc.normal_derivative[:, source],
                                        max(maximum(abs, r.bem_neumann), floatmin(Float64)))
                                    print_remainder_error(stdout, "$name excitation $column remainder pressure", ep, "Pa")
                                    print_remainder_error(stdout, "$name excitation $column remainder normal derivative", en, "Pa/m")
                                    @test ep.relative_significant < 1e-10 && en.relative_significant < 1e-10
                                    @test ep.absolute_residue <= 1e-10 * ep.total_scale
                                    @test en.absolute_residue <= 1e-10 * en.total_scale
                                    @test ep.absolute <= 1e-10 * ep.total_scale && en.absolute <= 1e-10 * en.total_scale
                                    continue
                                end
                                ep = pointwise_complex_error(tr.pressure[:, source], tc.pressure[:, source])
                                en = pointwise_complex_error(tr.normal_derivative[:, source], tc.normal_derivative[:, source])
                                println("  $name excitation $column source $source (other=$include_other): ",
                                        "pressure $(ep.relative), normal derivative $(en.relative)",
                                        en.zero_count > 0 ? " ($(en.zero_count) exact zeros, max abs $(en.absolute_at_zero) Pa/m)" : "")
                                @test ep.relative < 1e-10 && ep.absolute_at_zero <= 1e-10 * max(ep.scale, floatmin(Float64))
                                @test en.relative < 1e-10 && en.absolute_at_zero <= 1e-10 * max(en.scale, floatmin(Float64))
                            end
                            # Without the remainder column, sources outside the interface (motion,
                            # prescribed BEM velocity) leave a gap; it must match the dense replay's.
                            @test isapprox(tc.reconstruction_error, tr.reconstruction_error; rtol=1e-9, atol=1e-12)
                            include_other && @test tc.reconstruction_error < 1e-9
                        end
                        # Source order and a zero interface flux are preserved.
                        reversed = interface_radiation_traces(external, c, reverse(ranges))
                        @test reversed.pressure ≈ interface_radiation_traces(external, c, ranges).pressure[:, end:-1:1]
                        zero_solution = merge(c, (interface_flux=zero.(c.interface_flux),
                            bem_pressure=zero.(c.bem_pressure), bem_neumann=zero.(c.bem_neumann)))
                        @test iszero(norm(interface_radiation_traces(external, zero_solution, ranges).pressure))
                        @test size(interface_radiation_traces(external, c, UnitRange{Int}[]).pressure) == (length(c.bem_pressure), 0)
                    end
                finally
                    release_condensed_coupled_system!(dense)
                    release_condensed_coupled_system!(external)
                end
            end
        end

        @testset "unsupported combinations are rejected" begin
            solver = dense_hook_solver(Ref(0), Any[])
            # The replay needs the hook's exterior solve; a hook without one is rejected at build time.
            @test_throws ErrorException build(500.0; bem_solver=dense_hook_solver(Ref(0), Any[]; exterior_solve=false),
                                              retain_interface_radiation=true)
            @test_throws ErrorException build(500.0; bem_solver=solver, bem_operators=() -> nothing)
        end
    end
end
