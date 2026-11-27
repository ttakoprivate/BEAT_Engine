# Roadmap step 6: the driver switch BLAB_COUPLED_BEM_SOLVER. A dense reference hook (the same contract
# test solver as tests/external_bem_solver_tests.jl) is registered under a test name and run through the
# real JSON driver on the tiny compiled coupled workload (Float64). Its outputs must match the default
# dense path for every quantity, and the switch must fail loudly for unknown names and non-condensed
# solves instead of falling back to the dense path. This checks the wiring, not the H-matrix accuracy.
module ExternalBemSolverDriverTests
using Test, LinearAlgebra
const ENGINE_DIR = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ENGINE_DIR, "BeatEngineCompiledDriver.jl"))
include(joinpath(ENGINE_DIR, "..", "julia_engine", "CompiledCoupledWorkload.jl"))

const CC = BeatEngineCoupledCondensed

function dense_reference_factory(calls::Ref{Int}, frequencies::Vector{Float64})
    return () -> begin
        solved = Float64[]
        hook = context -> begin
            calls[] += 1
            push!(solved, context.wavenumber)
            operators = CC._assemble_condensed_bem_operators(
                context.bem_mesh, context.prepared, context.wavenumber, context.singular_order)
            A, R, _ = CC._combine_condensed_bem_operators!(operators, context.prepared, context.wavenumber)
            C = -Matrix(R)
            (; flux_operator_mul=Q -> C * Q,
               exterior_solve=(let F = lu(A); Rhs -> F \ Rhs; end),
               factorize=(rest, layout) -> begin
                   M = Matrix(rest)
                   B = layout.bem_range
                   M[B, B] .+= A
                   M[B, layout.flux_range] .+= C * layout.bem_flux
                   isempty(layout.mechanical_range) || (M[B, layout.mechanical_range] .+= C * layout.bem_motion_flux)
                   lu(M)
               end)
        end
        diagnostics = () -> begin
            d = Dict{String,Any}("wavenumbers" => copy(solved))
            append!(frequencies, solved); empty!(solved)
            d
        end
        (; bem_solver=hook, diagnostics)
    end
end

decode(q) = copy(reinterpret(q["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64,
                            base64decode(q["values"]["content_base64"])))

function request64()
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    request["solver_options"]["precision"] = "float64"
    return request
end

@testset "BLAB_COUPLED_BEM_SOLVER selection (roadmap step 6)" begin
    CC.unregister_external_bem_solver!("test_dense")
    @test CC.external_bem_solver_from_env(; condensed=true, env=Dict{String,String}()) === nothing
    @test CC.external_bem_solver_from_env(; condensed=false, env=Dict("BLAB_COUPLED_BEM_SOLVER" => " Dense ")) === nothing
    @test CC.external_bem_solver_from_env(; condensed=true, env=Dict("BLAB_COUPLED_BEM_SOLVER" => "")) === nothing
    unknown = try
        CC.external_bem_solver_from_env(; condensed=true, env=Dict("BLAB_COUPLED_BEM_SOLVER" => "hmatrix")); ""
    catch err
        sprint(showerror, err)
    end
    @test occursin("no external BEM solver", unknown) && occursin("hmatrix_dev", unknown)
    @test_throws ArgumentError CC.register_external_bem_solver!("dense", () -> nothing)
    @test_throws ArgumentError CC.register_external_bem_solver!("  ", () -> nothing)
    CC.register_external_bem_solver!("Test_Dense", () -> (; bem_solver=identity, diagnostics=nothing))
    try
        selected = CC.external_bem_solver_from_env(; condensed=true, env=Dict("BLAB_COUPLED_BEM_SOLVER" => "TEST_DENSE"))
        @test selected.name == "test_dense" && selected.bem_solver === identity && selected.diagnostics === nothing
        @test_throws ErrorException CC.external_bem_solver_from_env(; condensed=false,
            env=Dict("BLAB_COUPLED_BEM_SOLVER" => "test_dense"))
        CC.register_external_bem_solver!("test_dense", () -> (; diagnostics=nothing))
        @test_throws ErrorException CC.external_bem_solver_from_env(; condensed=true,
            env=Dict("BLAB_COUPLED_BEM_SOLVER" => "test_dense"))
    finally
        CC.unregister_external_bem_solver!("test_dense")
    end
end

@testset "BLAB_COUPLED_BEM_SOLVER through the JSON driver (roadmap step 6)" begin
    request = request64()
    calls = Ref(0); frequencies = Float64[]
    CC.register_external_bem_solver!("test_dense", dense_reference_factory(calls, frequencies))
    try
        withenv(coupled_workload_environment(; mumps=false, defaults=:cpu)...) do
            dense = solve_coupled_workload(request)
            external = withenv("BLAB_COUPLED_BEM_SOLVER" => "test_dense") do
                solve_coupled_workload(request)
            end
            nfreq = length(request["frequencies_hz"])
            @test dense.outcome.solved_count == nfreq && external.outcome.solved_count == nfreq
            @test calls[] == nfreq && length(frequencies) == nfreq
            for (d, e) in zip(dense.results, external.results)
                @test !haskey(d["diagnostics"], "coupled_bem_solver")
                @test e["diagnostics"]["coupled_bem_solver"] == "test_dense"
                @test length(e["diagnostics"]["coupled_bem_solver_diagnostics"]["wavenumbers"]) == 1
                @test e["diagnostics"]["coupled_sweep_pipeline"] == false
                @test e["diagnostics"]["interface_elimination"] == "none"
                @test [q["id"] for q in d["quantities"]] == [q["id"] for q in e["quantities"]]
                for (qd, qe) in zip(d["quantities"], e["quantities"])
                    r = ComplexF64.(decode(qd)); c = ComplexF64.(decode(qe))
                    nz = .!iszero.(r)
                    rel = maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz]); init=0.0)
                    zero_abs = maximum(abs.(c[.!nz]); init=0.0)
                    println("  driver switch $(d["freq_hz"]) Hz $(qd["id"]) ($(qd["unit"])): ",
                            "max complex relative error vs dense Float64 $(rel), entries $(count(nz)), ",
                            "exact zeros $(count(!, nz)) with max abs $(zero_abs) $(qd["unit"])")
                    @test rel < 1e-8
                    @test zero_abs == 0.0
                end
            end
            # Non-condensed (monolithic) solves reject the switch instead of ignoring it.
            monolithic = request64()
            monolithic["solver_options"]["static_condensation"] = false
            @test_throws Exception withenv("BLAB_COUPLED_BEM_SOLVER" => "test_dense") do
                solve_request(monolithic)
            end
            @test_throws Exception withenv("BLAB_COUPLED_BEM_SOLVER" => "no_such_solver") do
                solve_request(request64())
            end
        end
    finally
        CC.unregister_external_bem_solver!("test_dense")
    end
end
end
