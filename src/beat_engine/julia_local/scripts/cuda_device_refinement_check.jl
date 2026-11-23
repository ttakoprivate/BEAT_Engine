# CUDA check for the componentwise acceptance (PR #15): SplitRefinedLU around a device-held
# ComplexF32 LU (`_cuda_refined_dense_lu!`), with BLAB_COUPLED_DENSE_SPILL off/on and
# BLAB_COUPLED_DENSE_COMPONENTWISE 0/1.
#
# 1. Synthetic systems: tolerance, refinement steps, fallback, and the max complex relative error
#    per component against the dense Float64 LU of the same stored matrix.
# 2. Repository fixture (femvolume.msh / exterior_conforming.msh, transducer), monolithic CUDA coupled
#    path in Float32: the same per variant, plus the physical outputs against the CPU Float64 build.
#
#   julia --project=<CUDA environment> src/beat_engine/julia_local/scripts/cuda_device_refinement_check.jl
include(joinpath(@__DIR__, "..", "src", "BeatEngineCore.jl")); using .BeatEngineCore
include(joinpath(@__DIR__, "..", "src", "BeatEngineCoupled.jl")); using .BeatEngineCoupled
using CUDA, LinearAlgebra, Random, Printf, StaticArrays
const C = BeatEngineCoupled
CUDA.functional() || error("no functional CUDA device")
println("device: ", CUDA.name(CUDA.device()), ", CUDA runtime ", CUDA.runtime_version())

# max complex relative error over the non-zero reference entries; exact zeros as absolute error
function pointwise(reference, candidate)
    r = ComplexF64.(vec(reference)); c = ComplexF64.(vec(candidate))
    nz = .!iszero.(r)
    rel = any(nz) ? maximum(abs.(c[nz] .- r[nz]) ./ abs.(r[nz])) : NaN
    return rel, count(.!nz), any(.!nz) ? maximum(abs.(c[.!nz])) : 0.0
end

const VARIANTS = [(spill, cw) for spill in ("off", "on") for cw in ("0", "1")]

function report(label, F, n, err)
    @printf("  %-26s n=%5d tol=%-10s (expected %.4e) steps=%d fallback=%s ω=%s  max rel err vs F64 LU: %.3e\n",
            label, n, isnan(F.componentwise_tolerance) ? "off" : @sprintf("%.4e", F.componentwise_tolerance),
            2 * (n + 1) * eps(Float64), F.iterations, F.fallback_reason === nothing ? "none" : F.fallback_reason,
            isnan(F.componentwise_backward_error) ? "-" : @sprintf("%.2e", F.componentwise_backward_error), err)
end

function device_case(name, A32::Matrix{ComplexF32}, rhs::Matrix{ComplexF64})
    println("\n== synthetic: $name")
    reference = lu(ComplexF64.(A32)) \ rhs
    for (spill, cw) in VARIANTS
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill, "BLAB_COUPLED_DENSE_COMPONENTWISE" => cw) do
            F = C._cuda_refined_dense_lu!(CuArray(A32))
            x = F \ rhs
            report("spill=$spill componentwise=$cw", F, size(A32, 1), pointwise(reference, x)[1])
            @assert (spill == "on") == (F.spill_path !== nothing)
            C.release_split_factor!(F)
        end
    end
end

function scaled(rng, n, k, scale, condition)
    u, _ = qr(randn(rng, ComplexF64, n, n)); v, _ = qr(randn(rng, ComplexF64, n, n))
    B = Matrix(u) * Diagonal(exp10.(range(0, -log10(condition); length=n))) * Matrix(v)'
    d = ones(n); d[1:k] .= scale
    A = ComplexF32.(B * Diagonal(d))
    exact = randn(rng, ComplexF64, n, 2); exact[1:k, :] ./= scale
    return A, ComplexF64.(A) * exact
end

rng = MersenneTwister(7)
device_case("unit-test system, n=60, 30 columns scaled 1e-10, κ=1e4", scaled(rng, 60, 30, 1e-10, 1e4)...)
device_case("n=4000, 400 columns scaled 1e-8, κ=1e3", scaled(rng, 4000, 400, 1e-8, 1e3)...)
let n = 4000
    A = randn(rng, ComplexF32, n, n) ./ Float32(sqrt(n)) + 2.0f0I
    device_case("n=4000, well scaled", A, randn(rng, ComplexF64, n, 3))
end

# ---- repository fixture, monolithic CUDA coupled path ----
const RECORD = Ref{Any}(nothing)
@eval C function _coupled_device_solve(factorization, d_rhs, accelerator)
    factorization isa SplitRefinedLU || return factorization \ d_rhs
    rhs = ComplexF64.(Array(d_rhs))
    host_solution = factorization \ rhs
    store = factorization.store isa SpilledMatrix ? _load_all(factorization.store) : copy(factorization.store)
    Main.RECORD[] = (; matrix=store, rhs, solution=host_solution, factorization)
    return accelerator.CuArray(eltype(d_rhs).(host_solution))
end

const ROOT = joinpath(@__DIR__, "..", "tests", "fixtures")
const FIELDS = (:fem_pressure, :bem_pressure, :interface_flux, :diaphragm_velocity, :voice_coil_current)
function setup(::Type{T}) where {T}
    fem = load_gmsh41_volume(joinpath(ROOT, "femvolume.msh"), T(0.001))
    bem = load_gmsh22_with_tags(joinpath(ROOT, "exterior_conforming.msh"), T(0.001))
    map = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    transducer = ElectrodynamicTransducer{T}("component:test", [radiator], T[1], [1], T[-1], SVector{3,T}(0, 0, 1),
        T(2), 1, T(6), T(0.0005), T(7), T(0.015), T(0.0005), T(1))
    return (; fem, bem, map, radiator, transducer)
end
function solve_fixture(s, ::Type{T}, f, backend) where {T}
    system = build_coupled_system(s.fem, s.bem, s.map, T(f), T(343), T(1.21); validation_diagnostics=false,
                                  bem_backend=backend, transducers=[s.transducer])
    excitations = [(kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(1, 0)),
                   (kind=:voltage, radiator_tag=0, transducer_index=1, amplitude=Complex{T}(0, 2))]
    try
        return solve_coupled_excitations(system, excitations)
    finally
        release_coupled_system!(system)
    end
end

s32 = setup(Float32); s64 = setup(Float64)
for f in (2000.0, 10000.0)
    println("\n== fixture $(f) Hz, monolithic CUDA coupled path, Float32")
    full64 = solve_fixture(s64, Float64, f, :cpu)
    for (spill, cw) in VARIANTS
        withenv("BLAB_COUPLED_DENSE_SPILL" => spill, "BLAB_COUPLED_DENSE_COMPONENTWISE" => cw) do
            RECORD[] = nothing
            solutions = solve_fixture(s32, Float32, f, :cuda)
            rec = RECORD[]
            rec === nothing && error("the CUDA path did not use SplitRefinedLU")
            reference = lu(ComplexF64.(rec.matrix)) \ rec.rhs
            report("spill=$spill componentwise=$cw", rec.factorization, size(rec.matrix, 1),
                   pointwise(reference, rec.solution)[1])
            for field in FIELDS
                rels = [pointwise(getproperty(full64[k], field), getproperty(solutions[k], field)) for k in eachindex(solutions)]
                @printf("    %-20s vs CPU Float64 build: max rel %.3e (exact zeros %d, abs %.2e)\n", field,
                        maximum(r[1] for r in rels), rels[1][2], maximum(r[3] for r in rels))
            end
        end
    end
end
println("\nDONE")
