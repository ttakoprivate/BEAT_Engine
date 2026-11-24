# Step 2 of docs/ROADMAP.md (display only): per-output complex relative solver-error estimate from
# the componentwise bound of `_dense_error_estimate`, and the "recompute in Float64" hint.
isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
using Test, LinearAlgebra, Random

const OE = BeatEngineCoupledCondensed

# Dense system like the flux-eliminated build: a Float32-like BEM block plus Float64 border
# rows/columns (the driver unknowns), optionally ill-conditioned and with small border rows.
function oe_fixture(rng, n, border; condition=1e3, border_scale=1.0)
    m = n - border
    bem = if condition > 1e3
        u, _ = qr(randn(rng, ComplexF64, m, m)); v, _ = qr(randn(rng, ComplexF64, m, m))
        ComplexF32.(Matrix(u) * Diagonal(exp10.(range(0, -log10(condition); length=m))) * Matrix(v)')
    else
        randn(rng, ComplexF32, m, m) ./ Float32(sqrt(m)) + 2.0f0I
    end
    dense = zeros(ComplexF64, n, n)
    dense[1:m, 1:m] = bem
    dense[:, (m + 1):n] = randn(rng, ComplexF64, n, border)
    dense[(m + 1):n, 1:m] = border_scale .* randn(rng, ComplexF64, border, m)
    return dense, m
end

function oe_split(dense, m)
    n = size(dense, 1)
    split = OE.SplitDenseMatrix(n, (m + 1):n, (m + 1):n)
    for j in 1:n, i in 1:n
        split[i, j] = dense[i, j]
    end
    return split
end

@testset "componentwise bound covers the true solver error of every entry" begin
    setprecision(BigFloat, 160) do
        smallest = Inf             # smallest bound / truth over entries with a passed check
        for condition in (1e3, 1e5, 1e6), border_scale in (1.0, 1e-3), seed in 1:3
            rng = MersenneTwister(100 + seed)
            dense, m = oe_fixture(rng, 100, 8; condition, border_scale)
            rhs = randn(rng, ComplexF64, size(dense, 1), 3)
            exact = Complex{BigFloat}.(dense) \ Complex{BigFloat}.(rhs)
            for factorization in (OE.SplitRefinedLU(oe_split(dense, m)), OE.RefinedDenseLU(copy(dense)))
                solution = factorization \ rhs
                bound = factorization.error_bound
                @test size(bound) == size(solution)
                rows = (m + 1):size(dense, 1)
                for j in axes(solution, 2)
                    factorization.estimate_direction_check[j] || continue
                    truth = Float64.(abs.(Complex{BigFloat}.(solution[:, j]) .- exact[:, j]))
                    @test all(bound[:, j] .>= truth)
                    positive = truth .> 0
                    any(positive) && (smallest = min(smallest, minimum(bound[positive, j] ./ truth[positive])))
                    relative = OE._output_error_bound(bound[:, j], solution[:, j], rows)
                    @test all(relative .>= truth[rows] ./ abs.(solution[rows, j]))
                end
            end
        end
        @info "smallest componentwise bound / true error" smallest
        @test smallest >= 1
    end
end

@testset "output estimates per field and excitation, diagnostics and hint" begin
    rng = MersenneTwister(7)
    dense, m = oe_fixture(rng, 80, 6)
    rhs = randn(rng, ComplexF64, size(dense, 1), 2)
    for factorization in (OE.SplitRefinedLU(oe_split(dense, m)), OE.RefinedDenseLU(copy(dense)))
        solution = factorization \ rhs
        mechanical = (m + 1):(m + 3); electrical = (m + 4):(m + 6)
        system = (factorization=factorization, mechanical_range=mechanical, electrical_range=electrical,
                  scalar_type=Float32)
        OE._store_output_estimates!(system, solution)
        estimates = factorization.output_estimates
        @test Set(keys(estimates)) == Set([:diaphragm_velocity, :voice_coil_current])
        @test size(estimates[:diaphragm_velocity]) == (3, 2)
        @test all(e -> isfinite(e) && e >= 0, estimates[:voice_coil_current])
        # outputs stored as ComplexF32 carry their rounding √2 u in the estimate
        @test all(>=(sqrt(2) * eps(Float32) / 2), estimates[:diaphragm_velocity])
        OE._store_output_estimates!(merge(system, (scalar_type=Float64,)), solution)
        @test all(factorization.output_estimates[:diaphragm_velocity] .< sqrt(2) * eps(Float32) / 2)
        OE._store_output_estimates!(system, solution)
        estimates = factorization.output_estimates
        diagnostics = withenv("BLAB_OUTPUT_ERROR_HINT" => nothing) do
            OE.dense_solver_diagnostics(system)
        end
        @test diagnostics["dense_output_error_hint_threshold"] == 1e-3
        @test diagnostics["dense_output_estimated_relative_error_diaphragm_velocity"][2] ==
              estimates[:diaphragm_velocity][:, 2]
        # well-conditioned: refined to Float64 level, no hint
        @test diagnostics["dense_output_recompute_float64"] === false
        # a threshold below the estimate raises the hint for that entry
        tiny = minimum(estimates[:voice_coil_current]) / 2
        hinted = withenv("BLAB_OUTPUT_ERROR_HINT" => string(tiny)) do
            OE.dense_solver_diagnostics(system)
        end
        @test hinted["dense_output_recompute_float64"] === true
        @test all(hinted["dense_output_recompute_float64_voice_coil_current"][1])
        @test_throws ArgumentError withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => "-1")
        # a new solve invalidates the stored per-output estimates
        factorization \ rhs
        @test isnothing(factorization.output_estimates)
        @test !haskey(OE.dense_solver_diagnostics(system), "dense_output_recompute_float64")
    end
end

@testset "complex error: equal level with flipped phase raises the hint" begin
    # +1 Pa against -1 Pa: 0 dB level difference, complex relative error 2
    reference = ComplexF64[1, 0.5im]; candidate = -reference
    bound = abs.(candidate .- reference)                 # an exact componentwise bound of that error
    relative = OE._output_error_bound(bound, candidate, eachindex(candidate))
    @test relative ≈ [2.0, 2.0]
    @test all(20 .* log10.(abs.(candidate) ./ abs.(reference)) .== 0)
    @test all(OE.output_recompute_hint.(relative, 1e-3))
    @test all(OE.output_recompute_hint.(relative, 1.0))  # even a lax threshold below 200 %
    # unknown or infinite estimates always ask for the Float64 recompute
    @test OE.output_recompute_hint(NaN, 1.0) && OE.output_recompute_hint(Inf, 1.0)
    @test !OE.output_recompute_hint(1e-6, 1e-3)
    # zero entry: Inf with a non-zero bound, 0 with a zero bound; NaN bound stays unknown
    @test OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[1] == Inf
    @test OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[2] == 0.0
    @test isnan(OE._output_error_bound([1e-9, 0.0, NaN], ComplexF64[0, 0, 1], 1:3)[3])
end

@testset "failed direction check leaves the componentwise bound unknown and raises the hint" begin
    # review case 2 of PR #2: a Float32 pivot shifted by rounding, the direction check fails
    for tail in (2.0^-28, 2.0^-34)
        a = 1 + 2.0^-24
        A = ComplexF64[1 a; 1 (a + tail)]
        b = reshape((1 + im) .* [1, 1 + eps()], 2, 1)
        for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
            x = factorization \ b
            @test !factorization.estimate_direction_check[1]
            @test all(isnan, factorization.error_bound[:, 1])
            relative = OE._output_error_bound(factorization.error_bound[:, 1], x[:, 1], 1:2)
            @test all(OE.output_recompute_hint.(relative, 1.0))
        end
    end
end

@testset "REVIEW-22-b5ea266 P1: residual rounding is bounded, not taken from the correction" begin
    # nearly singular 2x2 with a scaled second row; the residual of the converged solve is pure
    # rounding, so the former σ ‖d‖ term underestimated entry 2 by about 90x without a hint
    rng = MersenneTwister(1)
    row = randn(rng, ComplexF64, 2)
    A = vcat(transpose(row), transpose(row + 1e-4 * randn(rng, ComplexF64, 2)))
    A[2, :] .*= 1e-3
    b = reshape(A[:, 1] + 7e-12 * A[:, 2], 2, 1)
    exact = setprecision(BigFloat, 256) do
        Complex{BigFloat}.(A) \ Complex{BigFloat}.(b)
    end
    for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
        x = factorization \ b
        truth = Float64.(abs.(Complex{BigFloat}.(x[:, 1]) .- exact[:, 1]))
        @test isnothing(factorization.fallback)
        @test all(b -> isnan(b) || b >= 0, factorization.error_bound)
        @test all(isnan.(factorization.error_bound[:, 1]) .| (factorization.error_bound[:, 1] .>= truth))
        system = (factorization=factorization, mechanical_range=2:2, electrical_range=1:1, scalar_type=Float32)
        OE._store_output_estimates!(system, x)
        diagnostics = withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => nothing)
        # the true relative error of entry 2 is several percent: the hint must be raised
        @test truth[2] / abs(x[2, 1]) > 1e-3
        @test diagnostics["dense_output_recompute_float64_diaphragm_velocity"][1][1] === true
        @test diagnostics["dense_output_recompute_float64"] === true
    end
end

@testset "REVIEW-22-b5ea266 P1: a residual rounded to zero does not give a zero bound" begin
    A = reshape(ComplexF64[3], 1, 1); b = reshape(ComplexF64[1], 1, 1)
    exact = setprecision(BigFloat, 256) do
        Complex{BigFloat}.(A) \ Complex{BigFloat}.(b)
    end
    for factorization in (OE.SplitRefinedLU(oe_split(A, 0)), OE.RefinedDenseLU(copy(A)))
        x = factorization \ b
        truth = Float64(abs(Complex{BigFloat}(x[1, 1]) - exact[1, 1]))
        @test truth > 0
        @test factorization.error_bound[1, 1] >= truth
    end
    # the estimator alone, with the reviewer's x = Float64(1/3) and its rounded-to-zero residual
    x = reshape(ComplexF64[1 / 3], 1, 1)
    residual = b .- A * x
    @test iszero(residual)
    rounding = OE._residual_rounding(b, abs.(A) * abs.(x), 1)
    _, passed, bound = OE._dense_error_estimate(r -> A \ r, y -> A * y, x, residual, b;
                                                apply_adjoint=r -> A' \ r, rounding)
    @test passed == [true]
    @test bound[1, 1] >= Float64(abs(big(1) / 3 - big(1 / 3)))
    # without an adjoint solve (device factor) the componentwise bound is unknown
    _, _, unknown = OE._dense_error_estimate(r -> A \ r, y -> A * y, x, residual, b)
    @test isnan(unknown[1, 1])
end

@testset "REVIEW-22-b5ea266 P2: SplitGMRES without fallback reports unknown outputs and the hint" begin
    A = ComplexF64[1 0; 0 1]
    solver = OE.SplitGMRES(oe_split(A, 0))
    x = solver \ reshape(ComplexF64[1, 2], 2, 1)
    @test isnothing(solver.fallback)
    system = (factorization=solver, mechanical_range=2:2, electrical_range=1:1, scalar_type=Float32)
    OE._store_output_estimates!(system, x)
    diagnostics = withenv(() -> OE.dense_solver_diagnostics(system), "BLAB_OUTPUT_ERROR_HINT" => nothing)
    @test diagnostics["dense_output_estimated_relative_error_diaphragm_velocity"] == [[nothing]]
    @test diagnostics["dense_output_estimated_relative_error_voice_coil_current"] == [[nothing]]
    @test diagnostics["dense_output_recompute_float64_diaphragm_velocity"] == [[true]]
    @test diagnostics["dense_output_recompute_float64"] === true
    solver \ reshape(ComplexF64[1, 2], 2, 1)          # a new solve invalidates them
    @test !haskey(OE.dense_solver_diagnostics(system), "dense_output_recompute_float64")
end

@testset "REVIEW-22-25b7064: documented limit, the norm estimator can miss a large mode" begin
    # Isolated estimator test from the review: a rank-one mode that none of the Hager-Higham search
    # directions (uniform start, unit vectors, alternating vector) excites. The estimate is a lower
    # bound and here ~4.2e6 times too small, so DENSE_NORM_ESTIMATE_MARGIN (3) is no guarantee and
    # the per-output bound stays a heuristic. This test pins the known limitation; if the estimator
    # is strengthened later it should be changed to require coverage.
    v = ComplexF64[0, 1, 1, -1, -1, 0, 0, 0]
    B = Matrix{ComplexF64}(I, 8, 8) + 2.0^20 * (v * v')
    f = ones(8, 1)
    estimate = OE.BeatEngineCoupled._inverse_scaled_norm(X -> B * X, X -> B' * X, f)[1]
    exact = opnorm(B, Inf)
    @test estimate <= exact * (1 + 1e-12)                      # still a lower bound
    @test exact == 1 + 4 * 2.0^20
    @test OE.BeatEngineCoupled.DENSE_NORM_ESTIMATE_MARGIN * estimate < exact / 1e6   # margin does not cover it
end

# Step 2, field points / polars: propagation of the boundary-data bounds through the field.
const OE_FIXTURES = joinpath(@__DIR__, "fixtures")
const OE_FIELD_POINTS = [SVector(sin(t), 0.0, cos(t)) for t in range(0, pi; length=7)]

@testset "field bound covers every boundary perturbation within the bounds" begin
    mesh = load_gmsh22_with_tags(joinpath(OE_FIXTURES, "exterior_conforming.msh"), 0.001)
    cache = build_field_evaluation_cache(mesh, triangle_rule(Float64, 2))
    rng = MersenneTwister(11)
    nodes = length(mesh.vertices); elements = length(mesh.faces)
    k = 2pi * 2000 / 343
    p = randn(rng, ComplexF64, nodes); q = randn(rng, ComplexF64, elements)
    pb = 1e-3 .* rand(rng, nodes) .* abs.(p); qb = 1e-3 .* rand(rng, elements) .* abs.(q)
    bound = galerkin_field_error_bound_cpu(OE_FIELD_POINTS, pb, qb, k, cache)
    base = evaluate_galerkin_field_cpu(OE_FIELD_POINTS, mesh, p, q, k, cache)
    scale = galerkin_field_error_bound_cpu(OE_FIELD_POINTS, abs.(p), abs.(q), k, cache)   # Σ|terms|
    for trial in 1:20
        # random phase, magnitude up to the bound; the last trial sits on the bound
        dp = (trial == 20 ? pb : pb .* rand(rng, nodes)) .* cis.(2pi .* rand(rng, nodes))
        dq = (trial == 20 ? qb : qb .* rand(rng, elements)) .* cis.(2pi .* rand(rng, elements))
        moved = evaluate_galerkin_field_cpu(OE_FIELD_POINTS, mesh, p + dp, q + dq, k, cache)
        @test all(abs.(moved - base) .<= bound .+ 1e-12 .* scale)
    end
    # a pressure bound on one node only: the field bound is the modulus of that node's contribution
    single = zeros(nodes); single[1] = 1.0
    unit = zeros(ComplexF64, nodes); unit[1] = 1.0
    contribution = abs.(evaluate_galerkin_field_cpu(OE_FIELD_POINTS, mesh, unit, zeros(ComplexF64, elements), k, cache))
    @test galerkin_field_error_bound_cpu(OE_FIELD_POINTS, single, zeros(elements), k, cache) >= contribution .* (1 - 1e-12)
    @test all(isnan, galerkin_field_error_bound_cpu(OE_FIELD_POINTS, fill(NaN, nodes), qb, k, cache))
    @test galerkin_field_error_bound_cpu(OE_FIELD_POINTS, zeros(nodes), zeros(elements), k, cache) == zeros(length(OE_FIELD_POINTS))
end

function oe_coupled(::Type{T}, f, env...) where {T}
    fem = load_gmsh41_volume(joinpath(OE_FIXTURES, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(OE_FIXTURES, "exterior_conforming.msh"), T(0.001))
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("component:test", [radiator], T[1], [1], T[-1], SVector{3,T}(0, 0, 1),
        T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    system = withenv(env...) do
        build_condensed_coupled_system(fem, bem, map, T(f), T(343), T(1.21); transducers=[transducer])
    end
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
                   (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(0, 2))]
    solutions = withenv(() -> solve_condensed_coupled_excitations(system, excitations), env...)
    return system, solutions, bem
end

@testset "field estimates on the 2-kHz fixture cover the solver error; unknown without a bound" begin
    common = ("BLAB_COUPLED_TRANSDUCER_CONDENSATION" => "1", "BLAB_COUPLED_INTERFACE_FLUX_ELIMINATION" => "1")
    reference, reference_solutions, bem = oe_coupled(Float32, 2000.0, common...,
        "BLAB_COUPLED_DENSE_REFINEMENT" => "0", "BLAB_COUPLED_DENSE_FLOAT64" => "1")
    cache64 = build_field_evaluation_cache(load_gmsh22_with_tags(joinpath(OE_FIXTURES, "exterior_conforming.msh"), 0.001),
                                           triangle_rule(Float64, reference.regular_quadrature_order))
    mesh64 = load_gmsh22_with_tags(joinpath(OE_FIXTURES, "exterior_conforming.msh"), 0.001)
    field64(x) = evaluate_galerkin_field_cpu(OE_FIELD_POINTS, mesh64, ComplexF64.(x.bem_pressure),
                                             ComplexF64.(x.bem_neumann), Float64(reference.wavenumber), cache64)
    try
        for (split, solver) in (("auto", "lu"), ("0", "lu"), ("auto", "gmres"))
            system, solutions, _ = oe_coupled(Float32, 2000.0, common..., "BLAB_COUPLED_DENSE_REFINEMENT" => "1",
                "BLAB_COUPLED_DENSE_FLOAT64" => "0", "BLAB_COUPLED_DENSE_SPLIT" => split, "BLAB_COUPLED_DENSE_SOLVER" => solver)
            try
                fields = [evaluate_galerkin_field_cpu(OE_FIELD_POINTS, bem, x.bem_pressure, x.bem_neumann,
                                                      system.wavenumber, system.field_cache) for x in solutions]
                estimates = OE.field_output_error_estimates(system, OE_FIELD_POINTS, fields, system.field_cache)
                @test size(estimates) == (length(OE_FIELD_POINTS), 2)
                gmres = system.factorization isa OE.SplitGMRES && isnothing(system.factorization.fallback)
                if gmres
                    @test all(isnan, estimates)          # no componentwise bound: unknown
                else
                    @test all(e -> isfinite(e) && e >= sqrt(2) * eps(Float32) / 2, estimates)
                    for j in 1:2
                        truth = abs.(field64(solutions[j]) - field64(reference_solutions[j])) ./
                                abs.(field64(reference_solutions[j]))
                        @test all(estimates[:, j] .>= truth)
                    end
                end
                @test_throws ArgumentError OE.field_output_error_estimates(system, OE_FIELD_POINTS, fields[1:1],
                                                                           system.field_cache)
                # a new solve of the factorization alone invalidates the stored bounds
                system.factorization \ ones(ComplexF64, size(system.factorization, 1))
                @test isnothing(OE.field_output_error_estimates(system, OE_FIELD_POINTS, fields, system.field_cache))
            finally
                release_condensed_coupled_system!(system)
            end
        end
    finally
        release_condensed_coupled_system!(reference)
    end
end
