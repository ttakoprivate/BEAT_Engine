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

complex_relative_error(reference, candidate) = begin
    r = ComplexF64.(reference); c = ComplexF64.(candidate)
    scale = maximum(abs, r; init=0.0)
    scale == 0 ? maximum(abs, c; init=0.0) : maximum(abs, c .- r; init=0.0) / scale
end

function compare_solutions(reference, candidate)
    worst = Dict{Symbol,Float64}()
    for name in propertynames(reference)
        value = getproperty(reference, name)
        value isa AbstractArray{<:Number} || continue
        eltype(value) <: Complex || eltype(value) <: AbstractFloat || continue
        hasproperty(candidate, name) || continue
        worst[name] = complex_relative_error(value, getproperty(candidate, name))
    end
    return worst
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
                    for (column, (r, c)) in enumerate(zip(reference, candidate))
                        worst = compare_solutions(r, c)
                        @test !isempty(worst)
                        for (output, err) in sort(collect(worst); by=first)
                            println("  $(case.name) excitation $column $(output): max complex relative error $(err)")
                            @test err < 1e-10
                        end
                    end
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
                    r = only(solve_condensed_coupled_excitations(dense, [voltage]))
                    c = only(solve_condensed_coupled_excitations(external, [voltage]))
                    for (output, err) in compare_solutions(r, c)
                        @test err < 1e-9
                    end
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
                @test complex_relative_error(dense.prescribed_bem_rhs, external.prescribed_bem_rhs) < 1e-12
            finally
                release_condensed_coupled_system!(dense)
                release_condensed_coupled_system!(external)
            end
        end

        @testset "unsupported combinations are rejected" begin
            solver = dense_hook_solver(Ref(0), Any[])
            @test_throws ErrorException build(500.0; bem_solver=solver, retain_interface_radiation=true)
            @test_throws ErrorException build(500.0; bem_solver=solver, bem_operators=() -> nothing)
        end
    end
end
