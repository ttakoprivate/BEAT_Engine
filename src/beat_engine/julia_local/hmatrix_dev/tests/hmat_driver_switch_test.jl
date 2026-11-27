# Roadmap step 6: `BLAB_COUPLED_BEM_SOLVER=hmatrix` through the real JSON driver. Loading
# hmatrix_dev/hmat_engine_solver.jl registers the H-matrix solver; the tiny compiled coupled workload
# (Float64) is then solved with the dense path and with the H-matrix path, and every output is compared
# pointwise (complex relative error) against the dense Float64 result. Also checks the per-frequency
# H-matrix diagnostics and the BLAB_HMATRIX_* settings. Run from src/beat_engine/julia_local with FMM3D,
# Krylov, LinearMaps, KernelAbstractions, Adapt on JULIA_LOAD_PATH:
#   julia --project=. hmatrix_dev/tests/hmat_driver_switch_test.jl
# A wiring and accuracy check on a tiny workload, not an h343 qualification and not a performance measurement.
module HMatDriverSwitchTest
using Test, LinearAlgebra
const JL_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ENGINE_DIR = JL_ROOT  # used by CompiledCoupledWorkload.jl
include(joinpath(JL_ROOT, "BeatEngineCompiledDriver.jl"))
include(joinpath(JL_ROOT, "..", "julia_engine", "CompiledCoupledWorkload.jl"))
include(joinpath(JL_ROOT, "src", "BeatEngineFmm.jl")); using .BeatEngineFmm
include(joinpath(JL_ROOT, "hmatrix_dev", "hmat_engine_solver.jl"))

# Same bound as hmat_engine_hook_test.jl (HMAT_TOL 1e-6, GMRES rtol 1e-9), pointwise for every output.
const SWITCH_BOUND = 1e-4

decode(q) = copy(reinterpret(q["values"]["dtype"] == "complex64" ? ComplexF32 : ComplexF64,
                            base64decode(q["values"]["content_base64"])))

function request64()
    request = JSON.parse(JSON.json(coupled_workload_request(; tiny=true)))
    request["solver_options"]["precision"] = "float64"
    return request
end

@testset "hmatrix settings from the environment" begin
    o = hmatrix_options_from_env(Dict{String,String}())
    @test o == (tol=1e-6, eta=1.0, leaf=48, rtol=1e-9, maxiter=300)
    o = hmatrix_options_from_env(Dict("BLAB_HMATRIX_TOL" => "1e-8", "BLAB_HMATRIX_LEAF" => "32", "BLAB_HMATRIX_RTOL" => " 1e-12 "))
    @test o.tol == 1e-8 && o.leaf == 32 && o.rtol == 1e-12 && o.eta == 1.0 && o.maxiter == 300
    for (name, value) in (("BLAB_HMATRIX_TOL", "0"), ("BLAB_HMATRIX_TOL", "abc"), ("BLAB_HMATRIX_ETA", "-1"),
                          ("BLAB_HMATRIX_LEAF", "1.5"), ("BLAB_HMATRIX_MAXITER", "0"), ("BLAB_HMATRIX_RTOL", "Inf"))
        @test_throws ArgumentError hmatrix_options_from_env(Dict(name => value))
    end
    @test haskey(BeatEngineCoupledCondensed._EXTERNAL_BEM_SOLVERS, "hmatrix")
end

@testset "BLAB_COUPLED_BEM_SOLVER=hmatrix through the JSON driver" begin
    request = request64()
    withenv(coupled_workload_environment(; mumps=false, defaults=:cpu)...) do
        dense = solve_coupled_workload(request)
        hmat = withenv("BLAB_COUPLED_BEM_SOLVER" => "hmatrix") do
            solve_coupled_workload(request)
        end
        nfreq = length(request["frequencies_hz"])
        @test dense.outcome.solved_count == nfreq && hmat.outcome.solved_count == nfreq
        for (d, h) in zip(dense.results, hmat.results)
            hd = h["diagnostics"]
            @test hd["coupled_bem_solver"] == "hmatrix"
            @test hd["interface_elimination"] == "none"
            @test hd["coupled_sweep_pipeline"] == false
            stats = hd["coupled_bem_solver_diagnostics"]
            @test stats["solver"] == "hmatrix"
            @test stats["build"]["tol"] == 1e-6 && 0 < stats["build"]["storage_A"] <= 1.0
            @test !isempty(stats["gmres_iterations"]) && all(stats["gmres_solved"])
            @test all(r -> r <= 1e-9, stats["gmres_relative_residuals"])
            println("  hmatrix switch $(d["freq_hz"]) Hz: GMRES iterations $(stats["gmres_iterations"]), ",
                    "residuals $(stats["gmres_relative_residuals"]), storage A $(stats["build"]["storage_A"]), ",
                    "storage C $(stats["build"]["storage_C"])")
            @test [q["id"] for q in d["quantities"]] == [q["id"] for q in h["quantities"]]
            for (qd, qh) in zip(d["quantities"], h["quantities"])
                r = ComplexF64.(decode(qd)); c = ComplexF64.(decode(qh))
                nz = .!iszero.(r)
                rel = maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz]); init=0.0)
                zero_abs = maximum(abs.(c[.!nz]); init=0.0)
                println("  hmatrix switch $(d["freq_hz"]) Hz $(qd["id"]) ($(qd["unit"])): max complex relative error ",
                        "vs dense Float64 $(rel), entries $(count(nz)), exact zeros $(count(!, nz)) with max abs ",
                        "$(zero_abs) $(qd["unit"])")
                @test rel < SWITCH_BOUND
                @test zero_abs <= SWITCH_BOUND * maximum(abs, r; init=0.0)
            end
        end
        # The tiny workload's BEM mesh has no admissible block pair even at leaf 8 (storage 1.0): this file
        # checks the switch, the settings and the JSON wiring. Compression accuracy against the dense Float64
        # path is covered on the larger fixture by hmat_engine_hook_test.jl.
        tight = withenv("BLAB_COUPLED_BEM_SOLVER" => "hmatrix", "BLAB_HMATRIX_TOL" => "1e-8",
                        "BLAB_HMATRIX_RTOL" => "1e-12", "BLAB_HMATRIX_LEAF" => "8") do
            solve_coupled_workload(request)
        end
        for r in tight.results
            build = r["diagnostics"]["coupled_bem_solver_diagnostics"]["build"]
            println("  hmatrix switch leaf 8 $(r["freq_hz"]) Hz: storage A $(build["storage_A"]), storage C $(build["storage_C"]), ",
                    "GMRES iterations $(r["diagnostics"]["coupled_bem_solver_diagnostics"]["gmres_iterations"])")
            @test build["tol"] == 1e-8 && build["leaf"] == 8 && build["rtol"] == 1e-12
            @test build["storage_A"] == 1.0  # no compression on this mesh, see above
        end
        for (d, t) in zip(dense.results, tight.results), (qd, qt) in zip(d["quantities"], t["quantities"])
            r = ComplexF64.(decode(qd)); c = ComplexF64.(decode(qt)); nz = .!iszero.(r)
            rel = maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz]); init=0.0)
            println("  hmatrix switch tol 1e-8 leaf 8 $(d["freq_hz"]) Hz $(qd["id"]): max complex relative error $(rel)")
            @test rel < 1e-6
        end
    end
end
end
