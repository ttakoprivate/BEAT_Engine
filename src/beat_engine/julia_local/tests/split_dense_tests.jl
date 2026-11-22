# Split-precision dense coupled storage (BeatEngineSplitDense.jl): exactness against the
# ComplexF64 matrix, equivalence with RefinedDenseLU, spilled residual store, GMRES-IR and the
# last-resort Float64 fallback.
isdefined(@__MODULE__, :BeatEngineCoupledCondensed) ||
    include(joinpath(@__DIR__, "coupled_condensed_test_setup.jl"))
using Test, LinearAlgebra, Random, JSON

const SD = BeatEngineCoupledCondensed

# A system laid out like the flux-eliminated dense build: ComplexF32 BEM block, Float64 border
# rows/columns, Float64 accumulations into a few "Γ-mapped" BEM columns.
function split_fixture(rng, n, border; condition=1e3)
    m = n - border
    bem = randn(rng, ComplexF32, m, m) ./ Float32(sqrt(m)) + 2.0f0I
    if condition > 1e3
        u, _ = qr(randn(rng, ComplexF64, m, m))
        v, _ = qr(randn(rng, ComplexF64, m, m))
        sigma = exp10.(range(0, -log10(condition); length=m))
        bem = ComplexF32.(Matrix(u) * Diagonal(sigma) * Matrix(v)')
    end
    gamma_columns = sort(randperm(rng, m)[1:max(1, m ÷ 10)])
    schur = randn(rng, ComplexF64, m, length(gamma_columns)) .* 1e-3
    border_cols = randn(rng, ComplexF64, n, border)
    border_rows = randn(rng, ComplexF64, border, m)
    return (; n, m, bem, gamma_columns, schur, border_cols, border_rows)
end

function split_fill!(target, f)
    m, n = f.m, f.n
    target[1:m, 1:m] = f.bem
    for (local_column, column) in enumerate(f.gamma_columns)
        @views target[1:m, column] .+= f.schur[:, local_column]
    end
    target[1:n, (m + 1):n] = f.border_cols
    target[(m + 1):n, 1:m] = f.border_rows
    @views target[(m + 1):n, f.gamma_columns[1]] .+= 1e-7          # an accumulation in a border row
    return target
end

split_build(f) = split_fill!(SD.SplitDenseMatrix(f.n, (f.m + 1):f.n, vcat((f.m + 1):f.n, f.gamma_columns)), f)
dense_build(f) = split_fill!(zeros(ComplexF64, f.n, f.n), f)

@testset "SplitDenseMatrix reproduces the ComplexF64 dense build exactly" begin
    rng = MersenneTwister(11)
    f = split_fixture(rng, 160, 12)
    split = split_build(f)
    dense = dense_build(f)
    @test Matrix(split) == dense                                   # bit-exact reconstruction
    @test split.hi == ComplexF32.(dense)                            # the factor RefinedDenseLU computes
    @test isempty(split.stray) && !split.overflow
    @test SD._split_opnorm_inf(split.hi, split) ≈ opnorm(dense, Inf) rtol = 1e-14
    x = randn(rng, ComplexF64, f.n, 3)
    y = zeros(ComplexF64, f.n, 3)
    SD._split_mul!(y, split.hi, split, x, one(ComplexF64), zero(ComplexF64))
    @test y ≈ dense * x rtol = 1e-13
    # A Float64 value outside the declared border is still stored exactly (as a stray).
    j = first(setdiff(1:f.m, f.gamma_columns))
    split[1, j] = 0.1
    @test split[1, j] === ComplexF64(0.1) && length(split.stray) == 1
    split[1, j] = 0.5                                               # representable: stray removed
    @test split[1, j] === ComplexF64(0.5) && isempty(split.stray)
end

@testset "SplitRefinedLU matches RefinedDenseLU" begin
    rng = MersenneTwister(12)
    f = split_fixture(rng, 220, 16)
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 4)
    reference = SD.RefinedDenseLU(copy(dense)) \ rhs
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            factorization = SD.SplitRefinedLU(split_build(f))
            @test (factorization.spill_path !== nothing) == (spill == "on")
            @test spill == "off" || isempty(factorization.matrix.hi)   # factored in place when spilled
            @test factorization.componentwise_tolerance == 2 * (f.n + 1) * eps(Float64)
            solution = factorization \ rhs
            @test norm(solution - reference) / norm(reference) < 1e-12
            @test factorization.backward_error <= 1 && isnothing(factorization.fallback)
            @test factorization.gmres_iterations == 0
            @test vec(factorization \ rhs[:, 1]) ≈ solution[:, 1] rtol = 1e-13
            @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_solver"] == "lu_float32_split_refined"
            @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_spill"] == (spill == "on")
        end
    end
end

@testset "GMRES-IR replaces the Float64 fallback when refinement stalls" begin
    rng = MersenneTwister(13)
    f = split_fixture(rng, 200, 10; condition=1e9)                  # κ eps(Float32) ≫ 1
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 2)
    exact = lu(dense) \ rhs
    # Both answers are Float64 backward stable; they may differ by up to ~κ eps(Float64).
    forward_tolerance = 100 * cond(dense, 1) * eps(Float64)
    @test cond(dense, 1) * eps(Float32) > 10                        # classical refinement cannot converge
    factorization = SD.SplitRefinedLU(split_build(f))
    solution = factorization \ rhs
    @test isnothing(factorization.fallback)
    @test factorization.gmres_iterations > 0
    @test factorization.backward_error <= 1
    @test norm(dense * solution - rhs) / (opnorm(dense, Inf) * norm(solution, Inf)) < 1e-13
    @test norm(solution - exact) / norm(exact) < forward_tolerance
    withenv("BLAB_COUPLED_DENSE_GMRES_IR" => "0") do               # old behaviour: Float64 LU
        legacy = SD.SplitRefinedLU(split_build(f))
        legacy_solution = (@test_logs (:warn,) match_mode=:any legacy \ rhs)
        @test !isnothing(legacy.fallback) && occursin("stalled", something(legacy.fallback_reason, ""))
        @test norm(legacy_solution - exact) / norm(exact) < forward_tolerance
    end
end

@testset "SplitRefinedLU falls back to Float64 for out-of-range entries" begin
    rng = MersenneTwister(14)
    f = split_fixture(rng, 120, 8)
    split = split_build(f)
    split[f.n, f.n] = 1.0e39                                        # beyond floatmax(Float32)
    @test split.overflow && split[f.n, f.n] == 1.0e39
    dense = Matrix(split)
    rhs = randn(rng, ComplexF64, f.n)
    factorization = @test_logs (:warn,) match_mode=:any SD.SplitRefinedLU(split)
    @test !isnothing(factorization.fallback) && occursin("Float32 range", factorization.fallback_reason)
    @test factorization \ rhs ≈ dense \ rhs rtol = 1e-10
    @test_throws ArgumentError SD.SplitRefinedLU(split_build(merge(f, (; bem=fill(NaN32 + 0im, f.m, f.m)))))
end

@testset "GMRES-IR rejects non-finite residuals after a finite RHS overflows Float32" begin
    # The original system is finite and exactly solvable. Converting the RHS for
    # the initial Float32 solve overflows; NaN residual ratios must not make the
    # GMRES-IR active set look converged. Include an already-solved finite column.
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill, "BLAB_COUPLED_DENSE_GMRES_IR" => "1") do
            for rhs in (ComplexF64[1e39, 2e39], ComplexF64[1 1e39; 2 2e39])
                split = SD.SplitDenseMatrix(2, Int[], Int[])
                split[1, 1] = 1
                split[2, 2] = 1
                factorization = SD.SplitRefinedLU(split)
                solution = (@test_logs (:warn,) match_mode=:any factorization \ rhs)
                @test size(solution) == size(rhs)
                @test all(isfinite, solution)
                @test solution == rhs
                @test !isnothing(factorization.fallback)
                @test occursin("non-finite residual", something(factorization.fallback_reason, ""))
                @test isfinite(factorization.backward_error) && factorization.backward_error <= 1
                @test factorization.gmres_iterations == 0
            end
        end
    end
end

@testset "SplitRefinedLU around a device-held factor (CUDA path, simulated on the host)" begin
    rng = MersenneTwister(15)
    n = 180
    matrix = randn(rng, ComplexF32, n, n) ./ Float32(sqrt(n)) + 2.0f0I
    rhs = randn(rng, ComplexF64, n, 3)
    reference = lu(ComplexF64.(matrix)) \ rhs
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            host = SD.SplitDenseMatrix(n, Int[], Int[])
            copyto!(host.hi, matrix)
            device_factor = lu(copy(matrix))                       # stands in for the CUDA LU
            solves = Ref(0)
            released = Ref(false)
            factorization = SD.SplitRefinedLU(host,
                x -> (solves[] += 1; ComplexF64.(device_factor \ x)), () -> (released[] = true; nothing))
            @test isnothing(factorization.factor) && (factorization.spill_path !== nothing) == (spill == "on")
            # review PR #15: the tolerance uses the system size, also after a spill empties `hi`
            @test factorization.componentwise_tolerance == 2 * (n + 1) * eps(Float64)
            solution = factorization \ rhs
            @test norm(solution - reference) / norm(reference) < 1e-12
            @test factorization.backward_error <= 1 && solves[] >= 2  # initial solve + refinement
            @test vec(factorization \ rhs[:, 2]) ≈ solution[:, 2] rtol = 1e-13
            SD.release_split_factor!(factorization)
            @test released[] && factorization.device_solve === nothing
            SD.release_split_factor!(factorization)                  # idempotent
        end
    end
end

@testset "dense LU library switch (BLAB_DENSE_LU_LIBRARY)" begin
    core = BeatEngineCore
    rng = MersenneTwister(7)
    for T in (ComplexF32, ComplexF64, Float64)
        A = randn(rng, T, 40, 40) + 8I
        withenv("BLAB_DENSE_LU_LIBRARY" => nothing) do
            @test core.dense_lu_library(T) == "blas"
            F = core.dense_lu!(copy(A))
            @test F.factors == lu(A).factors && F.ipiv == lu(A).ipiv
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "openblas") do
            @test core.dense_lu_library(T) == "blas"
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "auto") do
            library = core.dense_lu_library(T)
            @test library in ("blas", "accelerate")
            F = core.dense_lu!(copy(A))
            @test F isa LinearAlgebra.LU
            b = randn(rng, T, 40)
            @test norm(A * (F \ b) - b) <= 100 * eps(real(T)) * opnorm(A, 1) * norm(F \ b) * 40
        end
        withenv("BLAB_DENSE_LU_LIBRARY" => "accelerate") do
            if core._accelerate_getrf(T) == C_NULL
                @test_throws ErrorException core.dense_lu_library(T)
                @test startswith(core.dense_lu_library_diagnostic(T), "unavailable")
            else
                F = core.dense_lu!(copy(A))
                b = randn(rng, T, 40)
                @test norm(A * (F \ b) - b) <= 100 * eps(real(T)) * opnorm(A, 1) * norm(F \ b) * 40
                @test_throws SingularException core.dense_lu!(zeros(T, 4, 4))
                @test !issuccess(core.dense_lu!(zeros(T, 4, 4); check=false))
            end
        end
    end
    withenv("BLAB_DENSE_LU_LIBRARY" => "mkl") do
        @test_throws ErrorException core.dense_lu_library(ComplexF32)
    end
    @test core.dense_lu!(view(randn(rng, 6, 6) + 4I, :, :)) isa LinearAlgebra.LU   # non-Matrix: lu!
end

@testset "SplitGMRES (BLAB_COUPLED_DENSE_SOLVER=gmres): no factorization, falls back to LU" begin
    rng = MersenneTwister(21)
    f = split_fixture(rng, 200, 10)
    dense = dense_build(f)
    rhs = randn(rng, ComplexF64, f.n, 5)
    reference = dense \ rhs
    for precond in ("jacobi", "none")
        withenv("BLAB_COUPLED_DENSE_GMRES_PRECOND" => precond, "BLAB_COUPLED_DENSE_GMRES_TOL" => "1e-9") do
            solver = SD.SplitGMRES(split_build(f))
            solution = solver \ rhs
            @test isnothing(solver.fallback)
            @test solver.relative_residual <= 1e-9
            @test norm(solution - reference) / norm(reference) < 1e-6
            @test solver.inner_iterations > 0 && solver.outer_iterations >= 1
            d = SD.dense_solver_diagnostics((factorization=solver,))
            @test d["dense_solver"] == "gmres_float32_ir" && d["dense_gmres_preconditioner"] == precond
            @test vec(solver \ rhs[:, 2]) ≈ solution[:, 2] rtol = 1e-6
        end
    end
    withenv("BLAB_COUPLED_DENSE_GMRES_MAXITER" => "3") do              # budget too small: LU fallback
        solver = SD.SplitGMRES(split_build(f))
        solution = (@test_logs (:warn,) match_mode=:any solver \ rhs)
        @test solver.fallback isa SD.SplitRefinedLU
        @test norm(solution - reference) / norm(reference) < 1e-12
        diagnostics = SD.dense_solver_diagnostics((factorization=solver,))
        @test diagnostics["dense_solver"] == "gmres_fallback_lu"
        # the fallback LU's error estimate is passed through
        @test diagnostics["dense_estimated_relative_error_per_excitation"] == solver.fallback.estimated_error
        @test !isempty(solver.fallback.estimated_error) && diagnostics["dense_estimate_direction_check_passed"] === true
        # and its componentwise acceptance (Codex review on PR #15)
        @test diagnostics["dense_refinement_componentwise_tolerance"] == 2 * (f.n + 1) * eps(Float64)
        @test diagnostics["dense_refinement_componentwise_backward_error"] == solver.fallback.componentwise_backward_error
        @test diagnostics["dense_refinement_componentwise_backward_error"] <= diagnostics["dense_refinement_componentwise_tolerance"]
    end
end

@testset "SplitGMRES falls back when its Float32 correction is singular" begin
    for spill in ("off", "on")
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill) do
            # The FP64 border correction makes the exact system nonsingular,
            # but its Float32 surrogate annihilates this right-hand side.
            split = SD.SplitDenseMatrix(2, Int[], [2])
            split[1, 1] = split[1, 2] = split[2, 1] = 1
            split[2, 2] = 1 + 1e-10
            rhs = ComplexF64[1, -1]
            reference = Matrix(split) \ rhs
            solver = SD.SplitGMRES(split)
            solution = (@test_logs (:warn,) (:warn,) match_mode=:any solver \ rhs)
            @test all(isfinite, solution)
            @test solution ≈ reference rtol = 1e-12
            @test solver.fallback isa SD.SplitRefinedLU
            @test !isnothing(solver.fallback.fallback)
            @test occursin("singular projected system", something(solver.fallback_reason, ""))
        end
    end
end

@testset "SplitGMRES checks convergence after its final allowed correction" begin
    withenv("BLAB_COUPLED_DENSE_SPILL" => "off", "BLAB_COUPLED_DENSE_GMRES_MAXITER" => "1") do
        for rhs in (ComplexF64[1], ComplexF64[1 2])
            split = SD.SplitDenseMatrix(1, Int[], Int[])
            split[1, 1] = 1
            solver = SD.SplitGMRES(split)
            solution = (@test_logs solver \ rhs)
            @test solution == rhs
            @test isnothing(solver.fallback)
            @test solver.relative_residual == 0
            @test solver.inner_iterations == 1
            @test solver.outer_iterations == 1
        end
    end
end

@testset "Adopted A (BLAB_METAL_COUPLED_ADOPT_A): restrided in place, same system as a copy" begin
    rng = MersenneTwister(31)
    for (n, border) in ((160, 12), (97, 0), (64, 1))
        f = split_fixture(rng, n, border)
        m = f.m
        # Host memory laid out like the Metal assembly leaves it: A (m x m, leading dimension m)
        # first, then junk up to the n^2 capacity that must not leak into the system.
        address = Libc.malloc(n * n * sizeof(ComplexF32))
        raw = unsafe_wrap(Array, Ptr{ComplexF32}(address), n * n; own=false)
        fill!(raw, ComplexF32(NaN, NaN))
        copyto!(raw, 1, vec(f.bem), 1, m * m)
        hi = SD.adopt_square_storage(address, m, n)
        @test size(hi) == (n, n) && pointer(hi) == Ptr{ComplexF32}(address)
        adopted = SD.SplitDenseMatrix(hi, (m + 1):n, vcat((m + 1):n, f.gamma_columns))
        # Everything the build writes after the BEM block copy, which the adopted path skips.
        for (local_column, column) in enumerate(f.gamma_columns)
            @views adopted[1:m, column] .+= f.schur[:, local_column]
        end
        border > 0 && (adopted[1:n, (m + 1):n] = f.border_cols)
        border > 0 && (adopted[(m + 1):n, 1:m] = f.border_rows)
        border > 0 && (@views adopted[(m + 1):n, f.gamma_columns[1]] .+= 1e-7)
        @test Matrix(adopted) == dense_build(f)
        @test adopted.hi == split_build(f).hi
        rhs = randn(rng, ComplexF64, n, 2)
        @test SD.SplitRefinedLU(adopted) \ rhs ≈ dense_build(f) \ rhs rtol = 1e-10
    end
    @test_throws ArgumentError SD.adopt_square_storage(Ptr{Cvoid}(1), 5, 4)
end

@testset "a-posteriori error estimate tracks the true solver error" begin
    relative_error(x, exact) = [Float64(maximum(abs, x[:, j] - exact[:, j]) / maximum(abs, exact[:, j])) for j in axes(x, 2)]
    # every row and column on the Float64 border: the split matrix holds A exactly
    split_of(dense) = (n = size(dense, 1); m = SD.SplitDenseMatrix(n, 1:n, 1:n);
                       for j in 1:n, i in 1:n; m[i, j] = dense[i, j]; end; m)
    setprecision(BigFloat, 160) do
        for condition in (1e3, 1e6, 1e8)
            rng = MersenneTwister(21)
            f = split_fixture(rng, 100, 8; condition)
            dense = dense_build(f)
            rhs = randn(rng, ComplexF64, f.n, 3)
            exact = Complex{BigFloat}.(dense) \ Complex{BigFloat}.(rhs)
            for factorization in (SD.SplitRefinedLU(split_build(f)), SD.RefinedDenseLU(copy(dense)))
                solution = factorization \ rhs
                truth = relative_error(solution, exact)
                estimate = factorization.estimated_error
                passed = factorization.estimate_direction_check
                @test length(estimate) == 3 && all(isfinite, estimate)
                # κ eps(Float32) < 1: the Float32 factor is a good inverse and every column passes the direction check
                condition <= 1e6 && @test all(passed)
                # a column that passes the check is within a factor 10 of the true error on every path
                # (refined, GMRES-IR, Float64 fallback); at κ = 1e8 columns may fail it
                @test all(.!passed .| (0.1 .<= estimate ./ max.(truth, floatmin()) .<= 10))
                diagnostics = SD.dense_solver_diagnostics((factorization=factorization,))
                @test diagnostics["dense_estimated_relative_error"] == maximum(estimate)
                @test diagnostics["dense_estimated_relative_error_per_excitation"] == estimate
                @test diagnostics["dense_estimate_direction_check_passed"] === all(passed)
            end
        end
        # review case 1: a tiny excitation must not drop the estimate to zero (residual scaled before Float32)
        A = ComplexF64[1 0; (1 + eps()) 2.0^-25]
        s = 2.0^-99
        B = (1 + im) .* [1.0 s; 1.0 s]
        for factorization in (SD.SplitRefinedLU(split_of(A)), SD.RefinedDenseLU(copy(A)))
            factorization \ B
            e = factorization.estimated_error
            @test e[2] > 0 && isapprox(e[2], e[1]; rtol=1e-6)
        end
        # review case 2: a Float32 pivot shifted by rounding makes M⁻¹A far from I; the estimate
        # underestimates by 32x / 2048x here and the direction check must fail
        for tail in (2.0^-28, 2.0^-34)
            a = 1 + 2.0^-24
            A2 = ComplexF64[1 a; 1 (a + tail)]
            b = reshape((1 + im) .* [1, 1 + eps()], 2, 1)
            exact = Complex{BigFloat}.(A2) \ Complex{BigFloat}.(b)
            for factorization in (SD.SplitRefinedLU(split_of(A2)), SD.RefinedDenseLU(copy(A2)))
                x = factorization \ b
                truth = relative_error(x, exact)[1]
                ok = 0.1 <= factorization.estimated_error[1] / max(truth, floatmin()) <= 10
                @test ok || !factorization.estimate_direction_check[1]
                @test !factorization.estimate_direction_check[1]
            end
        end
    end
    # review 2, case 1: tiny but normal-range entries must give a finite estimate; non-finite ones are
    # reported as JSON null (the driver writes the diagnostics with JSON)
    for scale in (1.0, 2.0^-125)
        A = scale .* ComplexF64[1 -16; 0 1]
        b = reshape(scale * (1 + im) .* [1, 1 + eps()], 2, 1)
        exact = Complex{BigFloat}.(A) \ Complex{BigFloat}.(b)
        for factorization in (SD.SplitRefinedLU(split_of(A)), SD.RefinedDenseLU(copy(A)))
            x = factorization \ b
            e = factorization.estimated_error[1]
            truth = setprecision(() -> relative_error(x, exact)[1], BigFloat, 160)
            @test isfinite(e) && e <= max(10 * truth, 1e-15)
            text = JSON.json(SD.dense_solver_diagnostics((factorization=factorization,)))
            @test JSON.parse(text)["dense_estimated_relative_error"] == e
        end
    end
    for factorization in (SD.SplitRefinedLU(split_of(ComplexF64[1 0; 0 1])), SD.RefinedDenseLU(ComplexF64[1 0; 0 1]))
        factorization \ ComplexF64[1 1; 2.0^-1024 0]
        parsed = JSON.parse(JSON.json(SD.dense_solver_diagnostics((factorization=factorization,))))
        @test all(e -> e === nothing || isfinite(e), parsed["dense_estimated_relative_error_per_excitation"])
    end
    # review 3: a vanished solution (x = 0 for b = floatmin) must not report error 0
    for bval in (floatmin(Float64), 2.0^-1024)
        for factorization in (SD.SplitRefinedLU(split_of(reshape(ComplexF64[1], 1, 1))), SD.RefinedDenseLU(reshape(ComplexF64[1], 1, 1)))
            x = factorization \ ComplexF64[bval]
            parsed = JSON.parse(JSON.json(SD.dense_solver_diagnostics((factorization=factorization,))))
            if iszero(x[1])
                @test parsed["dense_estimated_relative_error_per_excitation"][1] == 1.0
                @test parsed["dense_estimate_direction_check_passed"] === false
            else
                @test parsed["dense_estimated_relative_error_per_excitation"][1] <= 1e-15
            end
        end
    end
    for factorization in (SD.SplitRefinedLU(split_of(reshape(ComplexF64[1], 1, 1))), SD.RefinedDenseLU(reshape(ComplexF64[1], 1, 1)))
        x = factorization \ ComplexF64[2.0^-1021]       # control: solved exactly
        @test x[1] == 2.0^-1021 && factorization.estimated_error[1] == 0 && factorization.estimate_direction_check[1]
    end
    # review 2, case 2: a decoupled third equation hides the shifted pivot of the first two from the
    # direction check (estimate 384x too small). Known limitation of the heuristic, documented in
    # _dense_error_estimate; a factorization-based safeguard is needed before automatic decisions.
    setprecision(BigFloat, 160) do
        a = 1 + 2.0^-24
        c = 2.0^-25
        A3 = ComplexF64[1 a 0; 1 (a + 2.0^-34) 0; 0 0 c]
        b3 = reshape((1 + im) .* [1, 1 + eps(), c * (1 + 2.0^-27)], 3, 1)
        # With the componentwise acceptance (issue #6) both solvers take one refinement step here
        # (the forward error stays at the Float64 level κ eps ≈ 4e-6). The split solver's residual
        # then happens to fail the direction check; the dense one still passes it with estimate 0.
        split3 = SD.SplitRefinedLU(split_of(A3)); split3 \ b3
        @test !split3.estimate_direction_check[1]
        dense3 = SD.RefinedDenseLU(copy(A3)); dense3 \ b3
        @test_broken !dense3.estimate_direction_check[1]
        withenv("BLAB_COUPLED_DENSE_COMPONENTWISE" => "0") do      # the original review case
            for factorization in (SD.SplitRefinedLU(split_of(A3)), SD.RefinedDenseLU(copy(A3)))
                factorization \ b3
                @test_broken !factorization.estimate_direction_check[1]
            end
        end
    end
    # cost: the estimate adds exactly one matrix product per solve, also on a reused factor
    rng = MersenneTwister(13)
    f = split_fixture(rng, 200, 8; condition=1e6)
    rhs = randn(rng, ComplexF64, f.n, 2)
    products(enabled) = withenv("BLAB_DENSE_ERROR_ESTIMATE" => enabled ? "1" : "0") do
        factorization = SD.SplitRefinedLU(split_build(f))
        counts = Int[]
        for b in (rhs, zero(rhs))
            before = parentmodule(SD.SplitRefinedLU).SPLIT_MUL_CALLS[]
            factorization \ b
            push!(counts, parentmodule(SD.SplitRefinedLU).SPLIT_MUL_CALLS[] - before)
        end
        counts
    end
    @test products(true) == products(false) .+ 1
    withenv("BLAB_DENSE_ERROR_ESTIMATE" => "0") do
        rng = MersenneTwister(1)
        f = split_fixture(rng, 60, 4)
        factorization = SD.SplitRefinedLU(split_build(f))
        factorization \ randn(rng, ComplexF64, f.n, 2)
        @test isempty(factorization.estimated_error)
        @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_estimated_relative_error"] === nothing
    end
end

@testset "componentwise acceptance: no silent error for badly scaled unknowns (issue #6)" begin
    split_of(dense) = (n = size(dense, 1); m = SD.SplitDenseMatrix(n, 1:n, 1:n);
                       for j in 1:n, i in 1:n; m[i, j] = dense[i, j]; end; m)
    # The criterion itself, on a given candidate (review example): A = diag(1, 2⁻⁶⁰),
    # b = (1, 2⁻⁶⁰), x = (1, 0). Normwise 0.002762 (accepted), componentwise ω = 1.
    A = ComplexF64[1 0; 0 2.0^-60]
    b = reshape(ComplexF64[1, 2.0^-60], 2, 1)
    x = reshape(ComplexF64[1, 0], 2, 1)
    r = b - A * x
    normwise = SD._dense_backward_error_ratios(SD.RefinedDenseLU(copy(A)), r, x)
    @test normwise[1] ≈ 0.002762 rtol = 1e-3
    omega = SD._componentwise_backward_errors(r, SD._dense_abs_product(A, x), b)
    @test omega == [1.0]
    ratios, _ = SD._acceptance_ratios(normwise, r, SD._dense_abs_product(A, x), b, SD._dense_componentwise_tolerance(2))
    @test ratios[1] > 1e14
    # zero row with zero residual contributes 0; a residual over a vanishing denominator is visible (Inf)
    @test SD._componentwise_backward_errors(zeros(ComplexF64, 2, 1), zeros(2, 1), zeros(ComplexF64, 2, 1)) == [0.0]
    @test SD._componentwise_backward_errors(reshape(ComplexF64[0, 1e-300], 2, 1), zeros(2, 1), zeros(ComplexF64, 2, 1)) == [Inf]
    @test SD._componentwise_backward_errors(reshape(ComplexF64[0, NaN], 2, 1), ones(2, 1), zeros(ComplexF64, 2, 1))[1] |> isnan

    # |A||x| from the split residual pass equals the dense one up to |hi| + |Δ| ≥ |hi + Δ|
    rng = MersenneTwister(31)
    f = split_fixture(rng, 160, 12)
    split = split_build(f)
    split[1, first(setdiff(1:f.m, f.gamma_columns))] = 0.1              # one stray entry
    dense = Matrix(split)
    xs = randn(rng, ComplexF64, f.n, 3)
    absolute = Matrix{Float64}(undef, f.n, 3)
    y = zeros(ComplexF64, f.n, 3)
    SD._split_mul!(y, split.hi, split, xs, one(ComplexF64), zero(ComplexF64); absolute)
    reference = abs.(dense) * abs.(xs)
    @test all(reference .* (1 - 1e-13) .<= absolute .<= reference .* (1 + 4 * eps(Float32)))
    @test y ≈ dense * xs rtol = 1e-13

    # Badly scaled unknowns (half the columns scaled by 1e-10, so those components are ~1e10 and
    # dominate ‖x‖∞): the normwise test accepts the plain Float32 solve with a 4e-4 error.
    function scaled_system(seed)
        rng = MersenneTwister(seed)
        n, k = 60, 30
        u, _ = qr(randn(rng, ComplexF64, n, n)); v, _ = qr(randn(rng, ComplexF64, n, n))
        B = Matrix(u) * Diagonal(exp10.(range(0, -4; length=n))) * Matrix(v)'
        d = ones(n); d[1:k] .= 1e-10
        exact = randn(rng, ComplexF64, n, 2); exact[1:k, :] ./= 1e-10
        A = B * Diagonal(d)
        return A, A * exact
    end
    relative(x, exact) = maximum(abs.(x .- exact) ./ abs.(exact))
    for seed in (2, 3)
        A, rhs = scaled_system(seed)
        reference = lu(A) \ rhs                                         # Float64 dense reference
        exact = setprecision(BigFloat, 160) do
            ComplexF64.(Complex{BigFloat}.(A) \ Complex{BigFloat}.(rhs))
        end
        for build in (() -> SD.SplitRefinedLU(split_of(A)), () -> SD.RefinedDenseLU(copy(A)))
            legacy = withenv(build, "BLAB_COUPLED_DENSE_COMPONENTWISE" => "0")
            legacy_x = legacy \ rhs
            @test legacy.iterations == 0                                # accepted unrefined ...
            @test relative(legacy_x, exact) > 1e-5                      # ... with a large error
            @test isnan(legacy.componentwise_tolerance)
            factorization = build()
            solution = factorization \ rhs
            @test isnothing(factorization.fallback) && factorization.iterations >= 2
            @test factorization.componentwise_backward_error <= factorization.componentwise_tolerance
            @test factorization.backward_error <= 1
            @test relative(solution, exact) < 1e-10                     # max complex relative error per component
            @test relative(solution, exact) <= 10 * relative(reference, exact) + 1e-12
            diagnostics = SD.dense_solver_diagnostics((factorization=factorization,))
            @test diagnostics["dense_refinement_componentwise_backward_error"] == factorization.componentwise_backward_error
            @test diagnostics["dense_refinement_componentwise_tolerance"] == 2 * (size(A, 1) + 1) * eps(Float64)
            parsed = JSON.parse(JSON.json(diagnostics))
            @test parsed["dense_refinement_componentwise_tolerance"] isa Real
            off = SD.dense_solver_diagnostics((factorization=legacy,))
            @test off["dense_refinement_componentwise_tolerance"] === nothing
        end
    end
    # GMRES-IR accepts only componentwise-converged columns too
    rng = MersenneTwister(13)
    g = split_fixture(rng, 200, 10; condition=1e9)
    factorization = SD.SplitRefinedLU(split_build(g))
    factorization \ randn(rng, ComplexF64, g.n, 2)
    @test isnothing(factorization.fallback) && factorization.gmres_iterations > 0
    @test factorization.componentwise_backward_error <= factorization.componentwise_tolerance
    # tolerance override and validation
    withenv("BLAB_COUPLED_DENSE_COMPONENTWISE_TOL" => "1e-9") do
        @test SD._dense_componentwise_tolerance(10) == 1e-9
    end
    withenv("BLAB_COUPLED_DENSE_COMPONENTWISE_TOL" => "-1") do
        @test_throws ErrorException SD._dense_componentwise_tolerance(10)
    end
end

@testset "Float64 fallback is refined to the componentwise tolerance (Codex review on PR #15)" begin
    split_of(dense) = (n = size(dense, 1); m = SD.SplitDenseMatrix(n, 1:n, 1:n);
                       for j in 1:n, i in 1:n; m[i, j] = dense[i, j]; end; m)
    # Wilkinson-type growth matrix (partial-pivoting growth 2⁵⁹): the Float64 LU solution alone has
    # ω ≈ 0.07 and relative error ≈ 1; scaled by 1e39 so that the Float32 path falls back at once.
    n = 60
    W = Matrix{ComplexF64}(I, n, n)
    for i in 2:n, j in 1:(i - 1); W[i, j] = -1; end
    W[:, n] .= 1
    W .*= 1e39
    rng = MersenneTwister(41)
    exact = randn(rng, ComplexF64, n, 2)
    rhs = W * exact
    plain = lu(W) \ rhs
    @test maximum(abs.(plain .- exact) ./ abs.(exact)) > 0.1                 # the unrefined fallback
    for build in (() -> SD.SplitRefinedLU(split_of(W)), () -> SD.RefinedDenseLU(copy(W)))
        factorization = @test_logs (:warn, r"outside the Float32 range") match_mode=:any build()
        @test !isnothing(factorization.fallback)
        solution = factorization \ rhs
        @test factorization.componentwise_backward_error <= factorization.componentwise_tolerance
        @test maximum(abs.(solution .- exact) ./ abs.(exact)) < 1e-12
        @test !occursin("above the tolerance", factorization.fallback_reason)
        @test factorization.fallback_refinement_steps == 1                     # review: steps are reported
        @test SD.dense_solver_diagnostics((factorization=factorization,))["dense_refinement_fallback_steps"] == 1
        @test vec(factorization \ rhs[:, 1]) ≈ solution[:, 1] rtol = 1e-12   # vector right-hand side
        # with the componentwise test off, the fallback solution is returned unrefined as before
        legacy = withenv(build, "BLAB_COUPLED_DENSE_COMPONENTWISE" => "0")
        @test maximum(abs.((legacy \ rhs) .- exact) ./ abs.(exact)) > 0.1
    end
end

@testset "Float64 fallback refinement keeps each right-hand side's best iterate (ChatGPT review on PR #15)" begin
    split_of(dense) = (n = size(dense, 1); m = SD.SplitDenseMatrix(n, 1:n, 1:n);
                       for j in 1:n, i in 1:n; m[i, j] = dense[i, j]; end; m)
    # review counterexample: two independent Wilkinson blocks (58, 129), one excitation each; the
    # refinement step helps the first column and not the second, which must not discard the first
    W(n) = ComplexF64[j == n || i == j ? 1 : i > j ? -1 : 0 for i in 1:n, j in 1:n]
    A = zeros(ComplexF64, 187, 187)
    A[1:58, 1:58] = W(58)
    A[59:end, 59:end] = W(129)
    rng = MersenneTwister(11)
    X = zeros(ComplexF64, 187, 2)
    X[1:58, 1] = complex.(rand(rng, 1:7, 58), rand(rng, 1:7, 58))
    X[59:end, 2] = complex.(rand(rng, -7:7, 129), rand(rng, -7:7, 129))
    B = A * X
    first_block = 1:58
    relative(x) = maximum(abs.(x[first_block] .- X[first_block, 1]) ./ abs.(X[first_block, 1]))
    for build in (() -> SD.SplitRefinedLU(split_of(A)), () -> SD.RefinedDenseLU(copy(A)))
        factorization = @test_logs (:warn,) match_mode=:any build()
        @test !isnothing(factorization.fallback)
        single = factorization \ B[:, 1]                       # converges alone: no warning
        batch = @test_logs (:warn,) match_mode=:any factorization \ B
        @test relative(single) < 1e-14
        @test relative(batch[:, 1]) < 1e-14                     # was 2.24 before the fix
        @test batch[:, 1] ≈ single rtol = 1e-14                 # same iterate alone and together
        @test iszero(batch[59:end, 1])                          # exact zeros stay exact
        # the reported ω belongs to the returned columns
        residual = B - A * batch
        omega = maximum(SD._componentwise_backward_errors(residual, abs.(A) * abs.(batch), B))
        @test factorization.componentwise_backward_error ≈ omega rtol = 1e-12
    end
end
