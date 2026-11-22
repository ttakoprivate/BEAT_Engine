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
peak(a, ref) = maximum(Indicator.relative_change.(a .- ref, ref); init=0.0)

function design(s, budget, seed)
    face = unique(s.interface_elimination_data.bem_of_gamma)
    free_surface = setdiff(s.bem_range, face)
    groups = (free_surface, face, s.mechanical_range, s.electrical_range)
    # Allocate the BEM budget equally to interface and exterior. All driver rows are included.
    counts = (min(cld(budget, 2), length(free_surface)), min(fld(budget, 2), length(face)),
              length(s.mechanical_range), length(s.electrical_range))
    return sample_equations(MersenneTwister(seed), size(s.factorization, 1), groups, counts)
end

function targeted_sample(mixed, ref, state, indices, seed, budget)
    # Choose certainty rows using only the mixed problem and the pilot targets.
    # |b| + |A||x| scales the adjoints across acoustic and driver equations.
    scale = abs.(state.rhs)
    a = mixed.factorization.matrix
    for j in eachindex(state.x), i in eachindex(scale)
        scale[i] += abs(a[i, j]) * abs(state.x[j])
    end
    bem = collect(mixed.bem_range)
    certain = Int[]
    for (field, index) in pairs(indices)
        g = Indicator.output_gradient(ref, field, index)
        dual = mixed.factorization.factor' \ ComplexF32.(g)
        scores = abs.(dual[bem]) .* scale[bem]
        append!(certain, bem[partialsortperm(scores, 1:2; rev=true)])
    end
    sort!(unique!(certain))
    rest = setdiff(bem, certain)
    groups = (certain, rest, mixed.mechanical_range, mixed.electrical_range)
    counts = (length(certain), budget-length(certain), length(mixed.mechanical_range), length(mixed.electrical_range))
    return sample_equations(MersenneTwister(seed), length(state.x), groups, counts)
end

function report_estimates(result, name, method, budget, seed, stored, truth)
    for field in physical_fields
        item = result.fields[field]
        actual = peak(getproperty(stored.fields, field), getproperty(truth.fields, field))
        @printf("ESTIMATE %s %s %d %d %s point=%d estimate=%.7g ratio=%.5g inconclusive=%s\n",
                name, method, budget, seed, field, item.index, item.relative, item.relative/actual, item.inconclusive)
        if method == "census"
            @test isapprox(item.relative, actual; rtol=0.005)
        end
    end
end

function calibrate()
    m32 = fixture(Float32)
    prep = @timed fixture(Float64)
    m64 = prep.value
    @printf("Float64 geometry and cache setup (once per mesh) %.5f s\n", prep.time)
    voltage = (kind=:voltage, transducer_index=1, amplitude=1+0im)
    velocity = (kind=:normal_velocity, radiator_tag=m64.radiator, amplitude=0.001+0.002im)
    println("Preparing the independently assembled 2 kHz models.")
    flush(stdout)
    mixed, full, ref = build(m32), build(m64), reference(m64)
    same_factor = lu(Matrix(mixed.factorization.matrix))
    same = merge(mixed, (factorization=same_factor,))
    println("LAYOUT ", mixed.interface_elimination, " n=", size(same_factor, 1))
    for (name, excitation) in (("voltage", voltage), ("velocity", velocity))
        state, stored, truth = capture(mixed, excitation), capture(same, excitation), capture(full, excitation)
        matrix_defect = truth.rhs - Indicator.factored_rows_product(full.factorization, eachindex(truth.x), state.x)
        for field in physical_fields
            y, ym, y64 = (getproperty(x.fields, field) for x in (state, stored, truth))
            @printf("REFERENCE %s %s total=%.7g assembly=%.7g solver=%.7g\n", name, field,
                    peak(y, y64), peak(ym, y64), peak(y, ym))
        end
        samples = [("random", budget, seed, design(mixed, budget, seed)) for budget in (16, 64) for seed in (11, 29, 47)]
        mixed_defect = state.rhs - mixed.factorization.matrix * state.x
        bem_defect = (matrix_defect - mixed_defect)[mixed.bem_range]
        @printf("PREFIX %s first64=%.7g all=%.7g\n", name, maximum(abs, bem_defect[1:64]), maximum(abs, bem_defect))
        # A full census tests signs and physical reconstruction independently of sampling variance.
        push!(samples, ("census", length(state.x), 0,
                        sample_equations(MersenneTwister(0), length(state.x), (eachindex(state.x),), (length(state.x),))))
        for (kind, budget, seed, sample) in samples
            result = assembly_indicator(mixed, ref, excitation, state, sample)
            observed = result.assembly_residual + result.mixed_residual
            agreement = maximum(abs, observed - matrix_defect[sample.rows])
            @test agreement <= 1e-10 * max(norm(truth.rhs, Inf), norm(state.x, Inf), 1e-8)
            report_estimates(result, name, kind, budget, seed, stored, truth)
            if kind == "random" && budget == 16
                # Freeze the pilot's candidate maxima, then use a fresh probability draw.
                targets = Dict(field => result.fields[field].index for field in physical_fields)
                validation = targeted_sample(mixed, ref, state, targets, seed+1000, 24)
                second = assembly_indicator(mixed, ref, excitation, state, validation; output_indices=targets)
                report_estimates(second, name, "independent", 40, seed, stored, truth)
            end
            @printf("ROW-CHECK %s %s %d %d %.7g\n", name, kind, budget, seed, agreement)
            flush(stdout)
        end
        repeated = capture(mixed, excitation)
        @test isequal(repeated.x, state.x)
        @test all(f -> isequal(getproperty(repeated.fields, f), getproperty(state.fields, f)), physical_fields)
    end
    measure_times(m32, m64, voltage)
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
            assembly_indicator(s, ref, voltage, state, design(s, 16, 11))
        end
        finished = time_ns()
        return (; total=(finished-started)*1e-9, solve=(solved-started)*1e-9,
                 reference=(prepared-solved)*1e-9, sample=(finished-prepared)*1e-9)
    end
    frequency_point(false)
    frequency_point(true)
    for repeat in 1:3, on in (false, true)
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
        if ARGS == ["--timing-only"]
            measure_times(fixture(Float32), fixture(Float64), (kind=:voltage, transducer_index=1, amplitude=1+0im))
        else
            calibrate()
        end
    end
end
