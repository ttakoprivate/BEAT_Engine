# Run from julia_local with julia --project=. --threads=1 scripts/calibrate_assembly_indicator.jl.
using LinearAlgebra, StaticArrays, Random, Printf, Test
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl"))
using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl"))
include(joinpath(@__DIR__, "..", "tests", "coupled_condensed_test_setup.jl"))
include(joinpath(@__DIR__, "..", "src", "BeatEngineAssemblyIndicator.jl"))
using .BeatEngineAssemblyIndicator
const Indicator = BeatEngineAssemblyIndicator
BLAS.set_num_threads(1)

function fixture(::Type{T}) where {T}
    fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), T(0.001))
    interface = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    driver = ElectrodynamicTransducer{T}("component:test", [radiator], [T(1)], [1], [-T(1)],
        SVector(T(0), T(0), T(1)), T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    vertices = sort(union(interface.fem_vertex_indices, findnz(assemble_transducer_operators(fem, bem, [driver]).fem_surface)[1]))
    cache = prepare_condensed_coupled_cache(fem, bem, interface; quadrature_order=1, singular_order=1,
                                            retained_fem_vertices=vertices, bem_backend=:cpu)
    return (; args=(fem, bem, interface, T(2000), T(343), T(1.21)),
             kwargs=(; quadrature_order=1, singular_order=1, transducers=[driver], cache, bem_backend=:cpu), radiator)
end

build(model) = build_condensed_coupled_system(model.args...; model.kwargs...)
reference(model) = build_indicator_reference(model.args...; model.kwargs...)
capture(s, e) = capture_coupled_state(b -> s.factorization \ b, s, e)
function field_error(a, ref)
    relative, absolute, zeros = 0.0, 0.0, 0
    for (x, y) in zip(a, ref)
        if iszero(y)
            absolute = max(absolute, abs(x))
            zeros += 1
        else
            relative = max(relative, abs(x - y) / abs(y))
        end
    end
    return (; relative, absolute, zeros)
end

function calibrate()
    m32, m64 = fixture(Float32), fixture(Float64)
    mixed, full, ref = build(m32), build(m64), reference(m64)
    same = merge(mixed, (; factorization=lu(Matrix(mixed.factorization.matrix))))
    voltage = (kind=:voltage, transducer_index=1, amplitude=1+0im)
    velocity = (kind=:normal_velocity, radiator_tag=m64.radiator, amplitude=0.001+0.002im)
    driver_fields = (:diaphragm_velocity, :voice_coil_current)
    for (name, excitation) in (("voltage", voltage), ("velocity", velocity))
        state, stored, truth = capture(mixed, excitation), capture(same, excitation), capture(full, excitation)
        for field in physical_fields
            y, ym, y64 = (getproperty(v.fields, field) for v in (state, stored, truth))
            total, assembly, solver = field_error(y, y64), field_error(ym, y64), field_error(y, ym)
            @printf("REFERENCE %s %s total=%.7g assembly=%.7g solver=%.7g zero_count=%d abs_at_zero=%.7g\n",
                    name, field, total.relative, assembly.relative, solver.relative, total.zeros, total.absolute)
            @test iszero(total.absolute)
        end
        n = length(state.x)
        census = sample_equations(MersenneTwister(0), n, (1:n,), (n,))
        exact = assembly_indicator(mixed, ref, excitation, state, census)
        @test Set(keys(exact.fields)) == Set(driver_fields)
        full_defect = truth.rhs - Indicator.factored_rows_product(full.factorization, 1:n, state.x)
        @test maximum(abs, exact.assembly_residual + exact.mixed_residual - full_defect) <=
              1e-10 * max(norm(truth.rhs, Inf), norm(state.x, Inf), 1e-8)
        for field in driver_fields
            true_delta = only(getproperty(truth.fields, field) - getproperty(stored.fields, field))
            @test abs(exact.fields[field].delta - true_delta) <= 0.005 * abs(true_delta)
        end
        # These seeds were held out when choosing the allocation and row budget.
        for seed in 101:116
            result = assembly_indicator(mixed, ref, excitation, state; rng=MersenneTwister(seed))
            @test length(intersect(result.rows, mixed.bem_range)) == 32
            @test result.assembly_residual ≈ exact.assembly_residual[result.rows] rtol=1e-6 atol=1e-13
            for field in driver_fields
                item = only(result.per_driver[field])
                @test item == result.fields[field]
                true_delta = only(getproperty(truth.fields, field) - getproperty(stored.fields, field))
                phase_error = abs(item.delta - true_delta) / abs(true_delta)
                ratio = abs(item.delta) / abs(true_delta)
                @test 1/3 <= ratio <= 3
                @printf("HELDOUT %s %s seed=%d ratio=%.7g complex_error=%.7g inconclusive=%s\n",
                        name, field, seed, ratio, phase_error, item.inconclusive)
            end
        end
        repeated = capture(mixed, excitation)
        @test isequal(repeated.x, state.x)
        @test all(f -> isequal(getproperty(repeated.fields, f), getproperty(state.fields, f)), physical_fields)
        @test_throws ArgumentError assembly_indicator(mixed, ref, excitation, state, census; output_indices=Dict(:bem_pressure=>1))
        current = assembly_indicator(mixed, ref, excitation, state, census; output_indices=Dict(:voice_coil_current=>1))
        @test only(keys(current.fields)) === :voice_coil_current
        @test current.fields[:voice_coil_current] == exact.fields[:voice_coil_current]
    end
end

function measure_times(m32, m64, voltage)
    function frequency_point(diagnostic)
        started = time_ns()
        s = build(m32)
        state = capture(s, voltage)
        solved = time_ns()
        prepared = solved
        if diagnostic
            ref = reference(m64)
            prepared = time_ns()
            assembly_indicator(s, ref, voltage, state; rng=MersenneTwister(101))
        end
        finished = time_ns()
        return (; total=(finished-started)*1e-9, solve=(solved-started)*1e-9,
                 reference=(prepared-solved)*1e-9, sample=(finished-prepared)*1e-9)
    end
    frequency_point(false)
    frequency_point(true)
    for repeat in 1:3, on in (isodd(repeat) ? (false, true) : (true, false))
        GC.gc()
        t = frequency_point(on)
        @printf("TIME %d %s %.6f solve=%.6f reference=%.6f sample=%.6f process_peak_MiB=%.1f\n",
                repeat, on ? "diagnostic" : "baseline", t.total, t.solve, t.reference, t.sample, Sys.maxrss()/2^20)
        flush(stdout)
    end
end

withenv("BLAB_COUPLED_DENSE_REFINEMENT"=>"1", "BLAB_COUPLED_DENSE_SPLIT"=>"1",
        "BLAB_COUPLED_DENSE_SPILL"=>"0", "BLAB_COUPLED_FEM_FLOAT64"=>"1",
        "BLAB_COUPLED_DENSE_SOLVER"=>"lu", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION"=>"1",
        "BLAB_COUPLED_TRANSDUCER_CONDENSATION"=>"1", "BLAB_COUPLED_FEM_SOLVER"=>"umfpack") do
    @testset "Independent 2 kHz assembly indicator calibration" begin
        ARGS in (String[], ["--accuracy-only"], ["--timing-only"]) || error("Expected --accuracy-only or --timing-only")
        if ARGS != ["--accuracy-only"]
            measure_times(fixture(Float32), fixture(Float64), (kind=:voltage, transducer_index=1, amplitude=1+0im))
        end
        if ARGS != ["--timing-only"]
            calibrate()
        end
    end
end
