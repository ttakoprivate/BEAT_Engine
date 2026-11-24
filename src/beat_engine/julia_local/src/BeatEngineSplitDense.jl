# Split-precision storage and refinement for the dense condensed coupled system.
#
# Included inside `BeatEngineCoupled` (shared by the condensed solver and the CUDA coupled path). Under `precision=float32` the refined dense path
# (`RefinedDenseLU`) used to hold the whole system twice: once in `ComplexF64` (the refinement
# residual) and once in `ComplexF32` (the LU factor), 24 bytes per entry. Most of the system is the
# BEM block `bem_lhs`, which arrives in `ComplexF32` and so carries no information that a
# `ComplexF64` copy adds. Only the rows and columns built from double-precision FEM quantities
# (Schur couplings, interface and transducer blocks) do.
#
# `SplitDenseMatrix` stores every entry as `ComplexF32` (`hi`) plus the exact `ComplexF64`
# remainder `m - ComplexF64(hi)` on a declared border of rows and columns. Writes accumulate in
# `ComplexF64` exactly as the dense `ComplexF64` matrix did, so `hi` is bit-identical to
# `ComplexF32.(m)` and the reconstructed matrix bit-identical to `m`. `SplitRefinedLU` then
# factors `hi` (the same factor `RefinedDenseLU` computed) and refines against the reconstructed
# matrix, so the two give the same answers up to the summation order of the residual product:
#
#   memory  RefinedDenseLU 24 n²  ->  SplitRefinedLU 16 n² (`hi` kept for the residual)
#                                  ->   8 n² with `BLAB_COUPLED_DENSE_SPILL`, which keeps the
#                                       residual copy of `hi` in an unlinked scratch file and
#                                       factors `hi` in place (the residual streams it from disk).
#
# A refinement that stalls continues as GMRES-IR (right-preconditioned GMRES in `ComplexF64`
# with the `ComplexF32` LU as preconditioner) instead of a `ComplexF64` LU, which would need
# 16 n² more. The `ComplexF64` LU stays as the last resort and says why it ran.

"""
`BLAB_COUPLED_DENSE_SPLIT=auto|1|0`: hold the refined dense coupled system as `ComplexF32` plus
exact `ComplexF64` border corrections (`SplitDenseMatrix`). `auto` (default) uses it wherever the
refined path runs; `0` restores the `ComplexF64` matrix of `RefinedDenseLU`.
"""
_dense_split_enabled() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_SPLIT", "auto"))) in ("auto", "1", "on", "true")

"""
`BLAB_COUPLED_DENSE_SPILL=off|auto|on|<directory>`: keep the refinement copy of the `ComplexF32`
system in a scratch file instead of RAM (8 n² instead of 16 n² resident). `auto` (default)
spills when 16 n² would exceed `BLAB_COUPLED_DENSE_SPILL_FRACTION` (default 0.5) of physical
memory; `on` always spills; a directory spills there (default: `tempdir()`).
"""
function _dense_spill_target(n::Integer)
    setting = strip(get(ENV, "BLAB_COUPLED_DENSE_SPILL", "auto"))
    lowered = lowercase(setting)
    lowered in ("off", "0", "false", "no") && return nothing
    if lowered == "auto"
        fraction = something(tryparse(Float64, get(ENV, "BLAB_COUPLED_DENSE_SPILL_FRACTION", "0.5")), 0.5)
        16.0 * float(n)^2 > fraction * Sys.total_memory() || return nothing
        return tempdir()
    end
    lowered in ("on", "1", "true", "yes") && return tempdir()
    isdir(setting) || error("BLAB_COUPLED_DENSE_SPILL directory does not exist: $setting")
    return setting
end

"""`BLAB_COUPLED_DENSE_GMRES_IR=1|0` (default 1): continue a stalled refinement as GMRES-IR."""
_dense_gmres_ir_enabled() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_GMRES_IR", "1"))) in ("1", "on", "true", "auto")

const SPLIT_REFINEMENT_MAX_ITERATIONS = 10       # as DENSE_REFINEMENT_MAX_ITERATIONS

# Componentwise acceptance (RefinedDenseLU and SplitRefinedLU).
#
# The normwise test `‖r‖∞ ≤ ‖x‖∞ ‖A‖∞ eps(Float64) √n` accepts a solution whose small components
# are wrong when rows or unknowns are badly scaled (interface flux ~1e4 next to pressure ~5): the
# residual of the small components drowns in `‖x‖∞ ‖A‖∞`. A solution is therefore accepted only if,
# per excitation j, also the componentwise (Oettli-Prager) backward error
#   ω_j = max_i |b - A x|ᵢⱼ / (|A| |xⱼ| + |bⱼ|)ᵢ
# is at most `_dense_componentwise_tolerance(n)`. A row with zero residual contributes zero; a
# non-zero residual over a vanishing denominator gives ω = Inf (never accepted). The default
# tolerance 2 (n + 1) eps(Float64) covers the rounding of the Float64 residual product itself
# (|computed - exact| ≤ γ_{n+1} (|A||x| + |b|)) and of x, so a solution at Float64 accuracy passes.
# Both tests are combined into one ratio per excitation, max(normwise ratio, ω / tolerance), which
# drives the existing convergence (≤ 1), stall (fails to halve) and iteration-limit rules.
# `|A| |x|` is accumulated in the residual pass over the same column panels (SplitRefinedLU) or
# in one extra pass over the Float64 matrix (RefinedDenseLU). For split storage `|hi| + |Δ|`
# replaces `|hi + Δ|` on the border, an upper bound within a factor 1 + 2 eps(Float32).
# Note: a small backward error does not imply a small forward error for an ill-conditioned system.
#   BLAB_COUPLED_DENSE_COMPONENTWISE=0      normwise test only (the previous behaviour)
#   BLAB_COUPLED_DENSE_COMPONENTWISE_TOL=ω  absolute tolerance on ω instead of 2 (n + 1) eps
_dense_componentwise_enabled() =
    !(lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_COMPONENTWISE", "1"))) in ("0", "off", "false", "no"))

function _dense_componentwise_tolerance(n::Integer)
    setting = strip(get(ENV, "BLAB_COUPLED_DENSE_COMPONENTWISE_TOL", ""))
    isempty(setting) && return 2.0 * (n + 1) * eps(Float64)
    tolerance = tryparse(Float64, setting)
    (tolerance === nothing || !(0 < tolerance < Inf)) &&
        error("BLAB_COUPLED_DENSE_COMPONENTWISE_TOL must be a positive number: $setting")
    return tolerance
end

# |z| without hypot's cost; falls back to abs() where abs2 under- or overflows
@inline function _abs_entry(z::Complex)
    a = sqrt(abs2(ComplexF64(z)))
    return (isfinite(a) && (a > 0 || iszero(z))) ? a : abs(ComplexF64(z))
end

"""Per column `max_i |residual|ᵢ / (absolute + |target|)ᵢ` (`absolute = |A| |x|`), see above."""
function _componentwise_backward_errors(residual::AbstractMatrix, absolute::AbstractMatrix, target::AbstractMatrix)
    errors = zeros(Float64, size(residual, 2))
    @inbounds for j in axes(residual, 2)
        worst = 0.0
        for i in axes(residual, 1)
            r = abs(residual[i, j])
            iszero(r) && continue
            denominator = absolute[i, j] + abs(target[i, j])
            worst = max(worst, denominator > 0 ? r / denominator : Inf)   # NaN propagates
        end
        errors[j] = worst
    end
    return errors
end

"""`|A| |x|` for a dense matrix, one column panel at a time (no n² temporary)."""
function _dense_abs_product(matrix::AbstractMatrix, x::AbstractMatrix)
    n = size(matrix, 1)
    width = max(1, min(size(matrix, 2), DENSE_SPLIT_PANEL_BYTES ÷ (16 * max(n, 1))))
    panel = Matrix{Float64}(undef, n, width)
    absolute_x = _abs_entry.(x)
    result = zeros(Float64, n, size(x, 2))
    for first_column in 1:width:size(matrix, 2)
        columns = first_column:min(first_column + width - 1, size(matrix, 2))
        block = view(panel, :, 1:length(columns))
        block .= _abs_entry.(view(matrix, :, columns))
        mul!(result, block, view(absolute_x, columns, :), 1.0, 1.0)
    end
    return result
end

"""
    _acceptance_ratios(normwise, residual, absolute, target, tolerance) -> (ratios, componentwise)

Per column `max(normwise, ω / tolerance)`; `absolute === nothing` (componentwise test off) returns
`normwise` and an empty `componentwise`.
"""
function _acceptance_ratios(normwise::Vector{Float64}, residual, absolute, target, tolerance::Float64)
    absolute === nothing && return normwise, Float64[]
    componentwise = _componentwise_backward_errors(residual, absolute, target)
    return [isnan(c) ? NaN : max(w, c / tolerance) for (w, c) in zip(normwise, componentwise)], componentwise
end

# A-posteriori solver-error estimate after an accepted solve (diagnostics only, the solution is
# not changed). From the final Float64 residual r:
#   d = M⁻¹ r                 (M: the factor the solver used; d ≈ x_exact - x when M⁻¹A ≈ I)
#   w = M⁻¹ (A d),  ρ = |w - d| / |d|      (how far M⁻¹A is from I in the direction of d)
# Per excitation the estimate is |d| / (1 - ρ) / |x| when ρ <= 1/2, else |d| / |x|.
# The flag `direction_check_passed` (ρ <= 1/2) is a cheap heuristic, not a guarantee: it tests M⁻¹A
# only in the direction of d. A badly rounded Float32 pivot in components that d hardly touches can
# pass it while the estimate is far too small (review case: 384x). It must not on its own release
# automatic decisions (e.g. skipping a Float64 recompute); that needs a factorization-based safeguard.
# Residual columns are scaled by a power of two chosen so that d is O(1) in the Float32 solve, so
# tiny or huge excitations neither underflow nor overflow; a non-finite estimate is stored as NaN,
# reported as `nothing` (JSON null) and fails the check. A solution that vanished for a non-zero
# excitation reports the exact relative error 1; a correction that underflowed although the residual
# does not vanish reports `nothing`; a residual of at least half the right-hand side fails the check.
# Cost: one matrix product and two solves with the existing factor.
# Scope: solver error for the assembled system, not discretization or assembly error.
# BLAB_DENSE_ERROR_ESTIMATE=0 skips it.
# After the Float64 LU fallback: up to FALLBACK_REFINEMENT_STEPS steps of fixed-precision refinement
# with that LU while the combined acceptance ratio (normwise and componentwise) exceeds 1 and
# decreases. One such step already makes a backward-stable LU solution componentwise stable under
# mild conditions (Skeel 1980); a large element growth can otherwise leave ω above the tolerance.
# A step that does not lower the ratio is discarded. If the ratio still exceeds 1 the solution is
# returned with a warning and a note in `fallback_reason` (the Float64 LU is the last resort; the
# achieved ω is in the diagnostics). With BLAB_COUPLED_DENSE_COMPONENTWISE=0 nothing changes. `acceptance!(x)` recomputes `residual` for x and returns
# (ratios, normwise); on return `residual` belongs to the returned solution.
const FALLBACK_REFINEMENT_STEPS = 2

function _refine_fallback!(acceptance!, fallback, solution, residual, factorization)
    ratios, normwise = acceptance!(solution)
    isnan(factorization.componentwise_tolerance) && return solution, normwise   # test off: previous behaviour
    # per right-hand side: refine only the open columns and keep each column's best iterate, so a
    # column that is not helped cannot discard the correction of an independent one (review PR #15)
    columns = solution isa AbstractVector ? reshape(solution, :, 1) : solution
    open = [isfinite(w) && r > 1 for (w, r) in zip(normwise, ratios)]
    best = copy(ratios)
    steps = 0
    while steps < FALLBACK_REFINEMENT_STEPS && any(open)
        correction = fallback \ (residual isa AbstractVector ? residual : residual[:, open])
        candidate = copy(columns)
        candidate[:, open] .+= correction isa AbstractVector ? reshape(correction, :, 1) : correction
        candidate_ratios, _ = acceptance!(solution isa AbstractVector ? vec(candidate) : candidate)
        improved = open .& (candidate_ratios .< best)
        if !any(improved)
            ratios, normwise = acceptance!(solution)          # restore the residual of the kept iterate
            break
        end
        columns[:, improved] .= candidate[:, improved]
        best[improved] .= candidate_ratios[improved]
        open .= improved .& (best .> 1)
        steps += 1
        # the residual must belong to the current iterate before the next correction or on return
        ratios, normwise = acceptance!(solution)
    end
    factorization.fallback_refinement_steps = max(factorization.fallback_refinement_steps, steps)
    if maximum(ratios; init=0.0) > 1
        note = "componentwise backward error $(factorization.componentwise_backward_error) above the tolerance " *
               "$(factorization.componentwise_tolerance) after the Float64 LU and $steps refinement step(s)"
        factorization.fallback_reason = something(factorization.fallback_reason, "Float64 LU") * "; " * note
        @warn note
    end
    return solution, normwise
end

const DENSE_ESTIMATE_MAX_CONTRACTION = 0.5
# Componentwise bound (see _dense_error_estimate). The rounding δr of the Float64 residual is
# bounded componentwise by f = γ (|b| + |A| |x|), γ = 2√2 (n + 2) eps(Float64) (complex inner
# products, worst case), and its effect ‖A⁻¹ δr‖_∞ <= ‖A⁻¹ diag(f)‖_∞ is estimated with the
# Hager-Higham 1-norm estimator on the adjoint (LAPACK's FERR approach) using the factor M, times
# DENSE_NORM_ESTIMATE_MARGIN for the estimator (a lower bound, normally within a factor 3) and
# 1/(1-ρ) for M⁻¹ vs A⁻¹. The margin is not a guarantee: a mode that none of the search directions
# excites can be missed entirely (review of 25b7064: rank-one mode, estimate 1 vs norm 4.2e6, pinned
# in output_error_estimate_tests.jl), so the bound and the hint stay heuristic and must not alone
# release automatic decisions. This replaces the former fixed σ ‖d‖, which a residual rounded to (near)
# zero defeats (review REVIEW-22-b5ea266, P1). Without an adjoint solve (device factor) the bound
# is unknown.
const DENSE_NORM_ESTIMATE_MARGIN = 3.0
const DENSE_NORM_ESTIMATE_ITERATIONS = 4
_dense_error_estimate_enabled() = !(lowercase(strip(get(ENV, "BLAB_DENSE_ERROR_ESTIMATE", "1"))) in ("0", "off", "false"))

_pow2_exponent(m) = (iszero(m) || !isfinite(m)) ? 0 : exponent(m)

# 2^k per column with |r| 2^k ≈ |b| / |x|, so that M⁻¹ (r 2^k) ≈ O(1) when M⁻¹ maps |b| to |x|.
_residual_scales(residual, target, solution) =
    [exp2(-_pow2_exponent(norm(view(residual, :, j), Inf)) + _pow2_exponent(norm(view(target, :, j), Inf)) -
          _pow2_exponent(norm(view(solution, :, j), Inf))) for j in axes(residual, 2)]

"""Componentwise bound `γ (|b| + |A| |x|)` of the rounding of a Float64 residual `b - A x` (n × n)."""
_residual_rounding(target, absolute, n) =
    (2 * sqrt(2.0) * (n + 2) * eps(Float64)) .* (_abs_entry.(target) .+ absolute)

"""
    _inverse_scaled_norm(apply, apply_adjoint, f) -> Vector{Float64}

Per column j an estimate (from below, Hager 1984 / Higham 1988) of `‖M⁻¹ diag(f_j)‖_∞`, i.e. of
`max_i (|M⁻¹| f_j)_i`, via the 1-norm of `diag(f_j) M⁻ᴴ`. Columns are scaled to max 1 inside so the
Float32 solves neither under- nor overflow. `NaN` where `f_j` is not finite.
"""
function _inverse_scaled_norm(apply, apply_adjoint, f::AbstractMatrix{Float64})
    n, k = size(f)
    scale = [maximum(view(f, :, j); init=0.0) for j in 1:k]
    result = fill(NaN, k)
    usable = [isfinite(s) for s in scale]
    any(usable) || return result
    g = f[:, usable] ./ transpose(max.(scale[usable], floatmin(Float64)))
    unit(z) = iszero(z) ? one(z) : z / abs(z)
    adjoint_product(X) = g .* apply_adjoint(X)          # C X with C = diag(g) M⁻ᴴ
    product(X) = apply(g .* X)                          # Cᴴ X
    best = zeros(size(g, 2))
    X = fill(ComplexF64(1 / n), n, size(g, 2))
    for iteration in 1:DENSE_NORM_ESTIMATE_ITERATIONS
        Y = adjoint_product(X)
        best .= max.(best, vec(sum(abs, Y; dims=1)))
        iteration == DENSE_NORM_ESTIMATE_ITERATIONS && break
        Z = product(unit.(Y))
        X = zeros(ComplexF64, n, size(g, 2))
        for j in axes(Z, 2)
            X[argmax(abs.(view(Z, :, j))), j] = 1
        end
    end
    alternating = [ComplexF64((-1)^(i + 1) * (1 + (i - 1) / max(n - 1, 1))) for i in 1:n]
    Y = adjoint_product(repeat(alternating, 1, size(g, 2)))
    best .= max.(best, 2 .* vec(sum(abs, Y; dims=1)) ./ (3n))
    result[usable] .= ifelse.(isfinite.(best), best, NaN) .* scale[usable]
    return result
end

"""
    _dense_error_estimate(apply, multiply, solution, residual, target; apply_adjoint, rounding)
        -> (estimates, direction_check_passed, componentwise_bound)

`apply(r)` applies the solver's factor M⁻¹, `multiply(y)` returns `A y` in Float64, `residual` is
the final residual `target - A solution`. See the comment above `DENSE_ESTIMATE_MAX_CONTRACTION`.
`apply_adjoint(r)` applies M⁻ᴴ and `rounding` is the componentwise bound of the residual rounding
(`_residual_rounding`); without either the componentwise bound is unknown (NaN).
"""
function _dense_error_estimate(apply, multiply, solution::AbstractMatrix, residual::AbstractMatrix,
                               target::AbstractMatrix; apply_adjoint=nothing, rounding=nothing)
    scales = _residual_scales(residual, target, solution)
    d = apply(residual .* transpose(scales))          # scaled d; unscaled below
    w = apply(multiply(d))
    # `rounding` may be a function, evaluated after `multiply` (which may compute |A| |x| in its pass)
    rounding = rounding isa Function ? rounding() : rounding
    rounding_norm = (isnothing(apply_adjoint) || isnothing(rounding)) ? fill(NaN, size(solution, 2)) :
                    _inverse_scaled_norm(apply, apply_adjoint, rounding)
    estimates = Float64[]; passed = Bool[]
    bound = fill(NaN, size(solution))                 # componentwise bound, see _output_error_bound
    for j in axes(solution, 2)
        rn = norm(view(residual, :, j), Inf); bn = norm(view(target, :, j), Inf)
        xn = norm(view(solution, :, j), Inf)
        if iszero(xn) && !iszero(bn)
            # the solution vanished for a non-zero excitation: the relative error is exactly 1
            push!(estimates, 1.0); push!(passed, false); continue
        end
        e = norm(view(d, :, j), Inf)
        if iszero(e) && !iszero(rn)
            # the correction underflowed although the residual does not vanish: unknown
            push!(estimates, NaN); push!(passed, false); continue
        end
        rho = iszero(e) ? 0.0 : norm(view(w, :, j) - view(d, :, j), Inf) / e
        x = max(xn, floatmin(Float64))
        ok = isfinite(rho) && rho <= DENSE_ESTIMATE_MAX_CONTRACTION
        estimate = (ok ? e / (1 - rho) : e) / scales[j] / x
        # a residual of the size of the right-hand side means the solution is not trustworthy either
        check = ok && isfinite(estimate) && !(rn >= bn / 2 && !iszero(rn))
        push!(passed, check)
        push!(estimates, isfinite(estimate) ? estimate : NaN)
        if check && isfinite(rounding_norm[j])
            # With r̂ = r + δr the computed residual (r = A err, err = x* - x) and E = I - M⁻¹A:
            #   err = (I - E)⁻¹ d - A⁻¹ δr,  (I - E)⁻¹ d - d = E (I - E)⁻¹ d, hence
            #   |err_i| <= |d_i| + ρ/(1-ρ) ‖d‖ + ‖A⁻¹ diag(f)‖,  ‖A⁻¹ diag(f)‖ <= ‖M⁻¹ diag(f)‖/(1-ρ).
            # ρ estimates ‖E‖ in the direction of d only and the norm is estimated (×margin), so the
            # bound inherits the heuristic caveat above. A failed check leaves NaN (unknown).
            propagated = DENSE_NORM_ESTIMATE_MARGIN * rounding_norm[j] / (1 - rho)
            bound[:, j] .= (abs.(view(d, :, j)) .+ rho / (1 - rho) * e) ./ scales[j] .+ propagated
        end
    end
    return estimates, passed, bound
end

"""
    _output_error_bound(bound, solution, rows) -> Vector{Float64}

Estimated complex relative solver error `|x*_i - x_i| / |x_i|` of the solution entries `rows` of
one column, from the componentwise `bound` of `_dense_error_estimate`. `NaN` (unknown) where the
bound is unknown; `Inf` where the entry is zero but its bound is not; `0` where both are zero.
"""
function _output_error_bound(bound::AbstractVector, solution::AbstractVector, rows)
    return [begin
                b = bound[i]; x = abs(ComplexF64(solution[i]))
                isnan(b) ? NaN : iszero(b) ? 0.0 : iszero(x) ? Inf : b / x
            end for i in rows]
end
const DENSE_GMRES_IR_RESTART = 30
const DENSE_GMRES_IR_MAX_CYCLES = 10
const DENSE_SPLIT_PANEL_BYTES = 256 * 1024^2     # ComplexF64 panel used by the residual product

"""
    SpilledMatrix(matrix)

A read-only `ComplexF32` matrix kept in an unlinked scratch file (column-major, streamed by
column panels). The open handle keeps the data alive; closing it (finalizer) frees the space.
No Mmap: the engine bundles do not declare that standard library.
"""
mutable struct SpilledMatrix <: AbstractMatrix{ComplexF32}
    io::IOStream
    rows::Int
    columns::Int
    path::String
    function SpilledMatrix(matrix::Matrix{ComplexF32}, directory::AbstractString)
        path, io = mktemp(directory; cleanup=true)
        write(io, matrix)
        flush(io)
        rm(path; force=true)                 # unlinked: the space returns when `io` is closed
        spilled = new(io, size(matrix, 1), size(matrix, 2), path)
        finalizer(spilled -> close(spilled.io), spilled)
        return spilled
    end
end

Base.size(matrix::SpilledMatrix) = (matrix.rows, matrix.columns)

function Base.getindex(matrix::SpilledMatrix, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    seek(matrix.io, ((j - 1) * matrix.rows + (i - 1)) * sizeof(ComplexF32))
    return read(matrix.io, ComplexF32)
end

"""Copy columns `columns` of `store` into `panel` (a ComplexF32 buffer of matching size)."""
_load_panel!(panel::AbstractMatrix{ComplexF32}, store::AbstractMatrix{ComplexF32}, columns) =
    copyto!(panel, view(store, :, columns))

function _load_panel!(panel::AbstractMatrix{ComplexF32}, store::SpilledMatrix, columns::UnitRange{Int})
    size(panel) == (store.rows, length(columns)) || throw(DimensionMismatch("panel size"))
    seek(store.io, (first(columns) - 1) * store.rows * sizeof(ComplexF32))
    if panel isa Matrix{ComplexF32}
        read!(store.io, panel)
    else
        buffer = Matrix{ComplexF32}(undef, size(panel))
        read!(store.io, buffer)
        copyto!(panel, buffer)
    end
    return panel
end

_load_all(store::AbstractMatrix{ComplexF32}) = store
function _load_all(store::SpilledMatrix)
    matrix = Matrix{ComplexF32}(undef, size(store))
    return _load_panel!(matrix, store, 1:store.columns)
end

"""
    SplitDenseMatrix(n, border_rows, border_columns)

`n × n` matrix whose entry `(i, j)` is `ComplexF64(hi[i, j]) + Δ(i, j)`. `Δ` lives in `dcol`
(every row of a border column), in `drow` (border rows outside the border columns), or, for a
correction found anywhere else, in `stray` (counted; expected empty). Entries whose value does not
fit `Float32` keep `hi = 0` and the whole value in `Δ` and set `overflow`.
"""
mutable struct SplitDenseMatrix <: AbstractMatrix{ComplexF64}
    hi::Matrix{ComplexF32}
    col_slot::Vector{Int32}
    row_slot::Vector{Int32}
    border_columns::Vector{Int}
    border_rows::Vector{Int}
    dcol::Matrix{ComplexF64}
    drow::Matrix{ComplexF64}
    stray::Dict{Tuple{Int,Int},ComplexF64}
    overflow::Bool
end

function SplitDenseMatrix(n::Integer, border_rows, border_columns)
    columns = sort!(unique!(collect(Int, border_columns)))
    rows = sort!(unique!(collect(Int, border_rows)))
    all(c -> 1 <= c <= n, columns) && all(r -> 1 <= r <= n, rows) || throw(ArgumentError("border index out of range"))
    col_slot = zeros(Int32, n)
    row_slot = zeros(Int32, n)
    for (slot, column) in enumerate(columns)
        col_slot[column] = slot
    end
    for (slot, row) in enumerate(rows)
        row_slot[row] = slot
    end
    return SplitDenseMatrix(
        zeros(ComplexF32, n, n), col_slot, row_slot, columns, rows,
        zeros(ComplexF64, n, length(columns)), zeros(ComplexF64, length(rows), n),
        Dict{Tuple{Int,Int},ComplexF64}(), false,
    )
end

"""
    SplitDenseMatrix(hi, border_rows, border_columns)

Split storage around an existing `ComplexF32` square `hi` (adopted, not copied).
"""
function SplitDenseMatrix(hi::Matrix{ComplexF32}, border_rows, border_columns)
    n = size(hi, 1)
    size(hi, 2) == n || throw(ArgumentError("adopted dense storage must be square"))
    matrix = SplitDenseMatrix(0, Int[], Int[])
    columns = sort!(unique!(collect(Int, border_columns)))
    rows = sort!(unique!(collect(Int, border_rows)))
    all(c -> 1 <= c <= n, columns) && all(r -> 1 <= r <= n, rows) || throw(ArgumentError("border index out of range"))
    col_slot = zeros(Int32, n)
    row_slot = zeros(Int32, n)
    for (slot, column) in enumerate(columns)
        col_slot[column] = slot
    end
    for (slot, row) in enumerate(rows)
        row_slot[row] = slot
    end
    return SplitDenseMatrix(
        hi, col_slot, row_slot, columns, rows,
        zeros(ComplexF64, n, length(columns)), zeros(ComplexF64, length(rows), n),
        Dict{Tuple{Int,Int},ComplexF64}(), false,
    )
end

"""
    adopt_square_storage(address, n, order) -> Matrix{ComplexF32}

Turn malloc'd memory holding an n x n `ComplexF32` matrix (leading dimension n) with room for
`order^2` elements into an `order x order` matrix whose top-left n x n block is that matrix and
whose remaining rows and columns are zero. In place: the columns move back to front (memmove),
so no second n x n copy ever exists. The returned array owns the memory (`free` at GC).
"""
function adopt_square_storage(address::Ptr, n::Int, order::Int)
    order >= n || throw(ArgumentError("adopted order $order is smaller than the matrix ($n)"))
    pointer = Ptr{ComplexF32}(address)
    bytes = sizeof(ComplexF32)
    if order > n
        for column in n:-1:2
            ccall(:memmove, Ptr{Cvoid}, (Ptr{Cvoid}, Ptr{Cvoid}, Csize_t),
                  pointer + (column - 1) * order * bytes, pointer + (column - 1) * n * bytes, n * bytes)
        end
    end
    matrix = unsafe_wrap(Array, pointer, (order, order); own=true)
    if order > n
        Threads.@threads for column in 1:order
            first = column <= n ? n + 1 : 1
            @inbounds fill!(view(matrix, first:order, column), zero(ComplexF32))
        end
    end
    return matrix
end

Base.size(matrix::SplitDenseMatrix) = size(matrix.hi)
Base.IndexStyle(::Type{SplitDenseMatrix}) = IndexCartesian()

@inline function _split_correction(matrix::SplitDenseMatrix, i::Int, j::Int)
    @inbounds column = matrix.col_slot[j]
    column > 0 && return @inbounds matrix.dcol[i, column]
    @inbounds row = matrix.row_slot[i]
    row > 0 && return @inbounds matrix.drow[row, j]
    isempty(matrix.stray) && return zero(ComplexF64)
    return get(matrix.stray, (i, j), zero(ComplexF64))
end

@inline function Base.getindex(matrix::SplitDenseMatrix, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    return ComplexF64(@inbounds matrix.hi[i, j]) + _split_correction(matrix, i, j)
end

@inline function _split_round(value::ComplexF64)
    rounded = ComplexF32(value)
    isfinite(rounded) || !isfinite(value) || return zero(ComplexF32), value, true
    # m - ComplexF64(hi) is exact (Sterbenz): hi is the rounding of m.
    return rounded, value - ComplexF64(rounded), false
end

@inline function _split_store!(matrix::SplitDenseMatrix, value::ComplexF64, i::Int, j::Int)
    rounded, correction, overflow = _split_round(value)
    overflow && (matrix.overflow = true)
    @inbounds matrix.hi[i, j] = rounded
    @inbounds column = matrix.col_slot[j]
    if column > 0
        @inbounds matrix.dcol[i, column] = correction
        return value
    end
    @inbounds row = matrix.row_slot[i]
    if row > 0
        @inbounds matrix.drow[row, j] = correction
        return value
    end
    if iszero(correction)
        isempty(matrix.stray) || delete!(matrix.stray, (i, j))
    else
        matrix.stray[(i, j)] = correction
    end
    return value
end

@inline function Base.setindex!(matrix::SplitDenseMatrix, value, i::Int, j::Int)
    @boundscheck checkbounds(matrix, i, j)
    _split_store!(matrix, ComplexF64(value), i, j)
    return matrix
end

# The BEM block arrives as ComplexF32: copy it straight into `hi`, clearing any corrections.
function Base.setindex!(matrix::SplitDenseMatrix, block::AbstractMatrix{ComplexF32},
                        rows::AbstractUnitRange{<:Integer}, columns::AbstractUnitRange{<:Integer})
    size(block) == (length(rows), length(columns)) || throw(DimensionMismatch("block size does not match the ranges"))
    copyto!(view(matrix.hi, rows, columns), block)
    for (slot, column) in enumerate(matrix.border_columns)
        column in columns && fill!(view(matrix.dcol, rows, slot), zero(ComplexF64))
    end
    for (slot, row) in enumerate(matrix.border_rows)
        row in rows && fill!(view(matrix.drow, slot, columns), zero(ComplexF64))
    end
    isempty(matrix.stray) || filter!(entry -> !(first(entry)[1] in rows && first(entry)[2] in columns), matrix.stray)
    return matrix
end

function Base.setindex!(matrix::SplitDenseMatrix, block::AbstractMatrix,
                        rows::AbstractVector{<:Integer}, columns::AbstractVector{<:Integer})
    size(block) == (length(rows), length(columns)) || throw(DimensionMismatch("block size does not match the indices"))
    @inbounds for (local_column, column) in enumerate(columns), (local_row, row) in enumerate(rows)
        _split_store!(matrix, ComplexF64(block[local_row, local_column]), Int(row), Int(column))
    end
    return matrix
end

"""Materialize the exact `ComplexF64` matrix (the last-resort fallback; 16 n²)."""
function _split_materialize(store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix)
    dense = ComplexF64.(_load_all(store))
    _split_add_corrections!(dense, matrix)
    return dense
end

function _split_add_corrections!(dense::AbstractMatrix{ComplexF64}, matrix::SplitDenseMatrix)
    for (slot, column) in enumerate(matrix.border_columns)
        @views dense[:, column] .+= matrix.dcol[:, slot]
    end
    for (slot, row) in enumerate(matrix.border_rows), column in axes(dense, 2)
        matrix.col_slot[column] == 0 && (dense[row, column] += matrix.drow[slot, column])
    end
    for ((i, j), correction) in matrix.stray
        dense[i, j] += correction
    end
    return dense
end

"""
    _split_mul!(result, store, matrix, x, alpha, beta)

`result = beta*result + alpha*M*x` in `ComplexF64`, where `M = ComplexF64.(store) + Δ(matrix)`.
`store` is the `ComplexF32` copy of `hi` (in RAM or memory-mapped); it is promoted one column panel
at a time, so the product streams `store` once.
"""
const SPLIT_MUL_CALLS = Threads.Atomic{Int}(0)    # full matrix streams; read by the tests

function _split_mul!(result::AbstractMatrix{ComplexF64}, store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix,
                     x::AbstractMatrix{ComplexF64}, alpha::ComplexF64, beta::ComplexF64;
                     absolute::Union{Nothing,Matrix{Float64}}=nothing)
    Threads.atomic_add!(SPLIT_MUL_CALLS, 1)
    n = size(store, 1)
    width = max(1, min(size(store, 2), DENSE_SPLIT_PANEL_BYTES ÷ (16 * n)))
    panel = Matrix{ComplexF64}(undef, n, width)
    panel32 = Matrix{ComplexF32}(undef, n, width)
    # `absolute` (optional) receives |M| |x| from the same panels (|hi| + |Δ| on the border)
    absolute_panel = absolute === nothing ? nothing : Matrix{Float64}(undef, n, width)
    absolute_x = absolute === nothing ? nothing : _abs_entry.(x)
    absolute === nothing || fill!(absolute, 0.0)
    iszero(beta) ? fill!(result, zero(ComplexF64)) : (isone(beta) || (result .*= beta))
    for first_column in 1:width:size(store, 2)
        last_column = min(first_column + width - 1, size(store, 2))
        columns = first_column:last_column
        loaded = length(columns) == width ? panel32 : Matrix{ComplexF32}(undef, n, length(columns))
        _load_panel!(loaded, store, columns)
        block = view(panel, :, 1:length(columns))
        block .= loaded
        mul!(result, block, view(x, columns, :), alpha, one(ComplexF64))
        if absolute !== nothing
            absolute_block = view(absolute_panel, :, 1:length(columns))
            absolute_block .= _abs_entry.(block)
            mul!(absolute, absolute_block, view(absolute_x, columns, :), 1.0, 1.0)
        end
    end
    # Border columns: every row; border rows: outside the border columns; strays.
    if !isempty(matrix.border_columns)
        mul!(result, matrix.dcol, x[matrix.border_columns, :], alpha, one(ComplexF64))
        absolute === nothing ||
            mul!(absolute, _abs_entry.(matrix.dcol), absolute_x[matrix.border_columns, :], 1.0, 1.0)
    end
    if !isempty(matrix.border_rows)
        row_part = matrix.drow * x                         # border-column entries of drow are zero
        absolute_row_part = absolute === nothing ? nothing : _abs_entry.(matrix.drow) * absolute_x
        for (slot, row) in enumerate(matrix.border_rows)
            @views result[row, :] .+= alpha .* row_part[slot, :]
            absolute === nothing || (@views absolute[row, :] .+= absolute_row_part[slot, :])
        end
    end
    for ((i, j), correction) in matrix.stray
        @views result[i, :] .+= (alpha * correction) .* x[j, :]
        absolute === nothing || (@views absolute[i, :] .+= _abs_entry(correction) .* absolute_x[j, :])
    end
    return result
end

"""Row-sum norm `‖M‖∞` of the reconstructed matrix."""
function _split_opnorm_inf(store::AbstractMatrix{ComplexF32}, matrix::SplitDenseMatrix)
    n = size(store, 1)
    sums = zeros(Float64, n)
    border = matrix.col_slot
    @inbounds for j in axes(store, 2)
        border[j] > 0 && continue
        for i in 1:n
            sums[i] += abs(ComplexF64(store[i, j]))
        end
    end
    # Border columns: |hi + Δ| in every row (drow is zero there).
    @inbounds for (slot, j) in enumerate(matrix.border_columns), i in 1:n
        sums[i] += abs(ComplexF64(store[i, j]) + matrix.dcol[i, slot])
    end
    # Border rows outside the border columns: replace |hi| by |hi + Δ|.
    @inbounds for (slot, i) in enumerate(matrix.border_rows), j in axes(store, 2)
        border[j] > 0 && continue
        correction = matrix.drow[slot, j]
        iszero(correction) && continue
        value = ComplexF64(store[i, j])
        sums[i] += abs(value + correction) - abs(value)
    end
    for ((i, j), correction) in matrix.stray
        value = ComplexF64(store[i, j])
        sums[i] += abs(value + correction) - abs(value)
    end
    return maximum(sums; init=0.0)
end

"""
    SplitRefinedLU(matrix::SplitDenseMatrix)

The refined dense solve of `RefinedDenseLU` over a `SplitDenseMatrix`: `ComplexF32` LU of `hi`,
iterative refinement against the exact matrix, GMRES-IR when refinement stalls, and the
`ComplexF64` LU (materialized) only as the last resort. `spill_path` names the memory-mapped file
holding the residual copy of `hi` when `BLAB_COUPLED_DENSE_SPILL` applies; it is unlinked as soon
as it is written, so its space returns when the factorization is collected.
"""
mutable struct SplitRefinedLU
    matrix::SplitDenseMatrix
    store::AbstractMatrix{ComplexF32}
    matrix_norm::Float64
    factor::Union{Nothing,LinearAlgebra.LU{ComplexF32,Matrix{ComplexF32},Vector{LinearAlgebra.BlasInt}}}
    fallback::Union{Nothing,LinearAlgebra.LU{ComplexF64,Matrix{ComplexF64},Vector{LinearAlgebra.BlasInt}}}
    iterations::Int
    gmres_iterations::Int
    fallback_reason::Union{Nothing,String}
    backward_error::Float64
    componentwise_backward_error::Float64     # worst column ω of the last solve (NaN when the test is off)
    componentwise_tolerance::Float64          # NaN when BLAB_COUPLED_DENSE_COMPONENTWISE=0
    estimated_error::Vector{Float64}          # per column of the last solve, see _dense_error_estimate
    estimate_direction_check::Vector{Bool}
    error_bound::Matrix{Float64}              # componentwise solver-error bound of the last solve
    output_estimates::Union{Nothing,Dict{Symbol,Matrix{Float64}}}   # set by the coupled solve
    spill_path::Union{Nothing,String}
    # A factor held elsewhere (the CUDA LU on the device): `device_solve(x::Matrix{ComplexF32})`
    # returns the ComplexF64 host solution, `release_device()` frees it. `factor` is then nothing.
    device_solve::Union{Nothing,Function}
    release_device::Union{Nothing,Function}
    fallback_refinement_steps::Int            # most Float64 refinement steps after the fallback (any solve)
end

_split_componentwise_tolerance(n) = _dense_componentwise_enabled() ? _dense_componentwise_tolerance(n) : NaN

function _split_spill(hi::Matrix{ComplexF32}, directory::AbstractString)
    store = SpilledMatrix(hi, directory)
    return store, store.path
end

function SplitRefinedLU(matrix::SplitDenseMatrix)
    all(isfinite, matrix.hi) && all(isfinite, matrix.dcol) && all(isfinite, matrix.drow) &&
        all(isfinite, values(matrix.stray)) ||
        throw(ArgumentError("dense coupled matrix has non-finite entries"))
    n = size(matrix, 1)
    spill_directory = _dense_spill_target(n)
    matrix_norm = _split_opnorm_inf(matrix.hi, matrix)
    store, spill_path = isnothing(spill_directory) ? (matrix.hi, nothing) : _split_spill(matrix.hi, spill_directory)
    refined = SplitRefinedLU(matrix, store, matrix_norm, nothing, nothing, 0, 0, nothing, NaN, NaN,
                             _split_componentwise_tolerance(n), Float64[], Bool[], zeros(0, 0), nothing, spill_path, nothing, nothing, 0)
    if matrix.overflow
        _split_fall_back!(refined, "an entry is outside the Float32 range")
        return refined
    end
    # With a spilled store `hi` is factored in place (8 n² resident); otherwise a copy is factored
    # and `hi` stays as the residual store (16 n²).
    working = isnothing(spill_path) ? copy(matrix.hi) : matrix.hi
    isnothing(spill_path) || (matrix.hi = Matrix{ComplexF32}(undef, 0, 0))
    candidate = dense_lu!(working; check=false)
    if issuccess(candidate) && all(isfinite, candidate.factors)
        refined.factor = candidate
    else
        _split_fall_back!(refined, "the Float32 factorization is singular or not finite")
    end
    return refined
end

"""
    SplitRefinedLU(matrix, device_solve, release_device)

Refinement on the host around a factorization held on a device (CUDA): `matrix.hi` is the
host copy of the matrix that was factored there (spilled to a scratch file under
`BLAB_COUPLED_DENSE_SPILL`, so the host may hold none of it). GMRES-IR needs the factors on the
host and is skipped; a stall falls back to a host `ComplexF64` LU.
"""
function SplitRefinedLU(matrix::SplitDenseMatrix, device_solve::Function, release_device::Function)
    all(isfinite, matrix.hi) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    n = size(matrix, 1)                  # before a spill empties `matrix.hi` (size reads it)
    spill_directory = _dense_spill_target(n)
    matrix_norm = _split_opnorm_inf(matrix.hi, matrix)
    store, spill_path = isnothing(spill_directory) ? (matrix.hi, nothing) : _split_spill(matrix.hi, spill_directory)
    isnothing(spill_path) || (matrix.hi = Matrix{ComplexF32}(undef, 0, 0))
    return SplitRefinedLU(matrix, store, matrix_norm, nothing, nothing, 0, 0, nothing, NaN, NaN,
                          _split_componentwise_tolerance(n), Float64[], Bool[], zeros(0, 0), nothing,
                          spill_path, device_solve, release_device, 0)
end

"""Free a device-held factor (no-op for host factors)."""
function release_split_factor!(factorization::SplitRefinedLU)
    release = factorization.release_device
    factorization.release_device = nothing
    factorization.device_solve = nothing
    release === nothing || release()
    return nothing
end

Base.size(factorization::SplitRefinedLU, dims...) = size(factorization.store, dims...)

function _split_fall_back!(factorization::SplitRefinedLU, reason::AbstractString)
    factorization.fallback_reason = "Float32 LU with refinement fell back to a Float64 LU: " * reason
    @warn factorization.fallback_reason * " (materializes the Float64 system: 16 n² bytes)"
    factorization.factor = nothing
    release_split_factor!(factorization)
    GC.gc()
    factorization.fallback = lu!(_split_materialize(factorization.store, factorization.matrix))
    return factorization
end

function _split_residual!(residual, factorization::SplitRefinedLU, target, solution; absolute=nothing)
    copyto!(residual, target)
    _split_mul!(residual, factorization.store, factorization.matrix, solution, -one(ComplexF64), one(ComplexF64);
                absolute)
    return residual
end

function _split_backward_error_ratios(factorization::SplitRefinedLU, residual, solution)
    threshold = factorization.matrix_norm * eps(Float64) * sqrt(size(factorization.store, 1))
    return [norm(view(residual, :, column), Inf) /
            max(norm(view(solution, :, column), Inf) * threshold, floatmin(Float64)) for column in axes(solution, 2)]
end

"""
    _split_acceptance!(residual, factorization, target, solution) -> (ratios, normwise)

Recomputes the residual (and `|A| |x|` in the same pass when the componentwise test is on) and
returns per column the combined acceptance ratio (≤ 1: accepted) and the normwise ratio alone.
Records the worst componentwise backward error in the factorization.
"""
function _split_acceptance!(residual, factorization::SplitRefinedLU, target, solution)
    absolute = isnan(factorization.componentwise_tolerance) ? nothing : Matrix{Float64}(undef, size(solution))
    _split_residual!(residual, factorization, target, solution; absolute)
    normwise = _split_backward_error_ratios(factorization, residual, solution)
    ratios, componentwise = _acceptance_ratios(normwise, residual, absolute, target, factorization.componentwise_tolerance)
    factorization.componentwise_backward_error = isempty(componentwise) ? NaN : maximum(componentwise)
    return ratios, normwise
end

function _split_precondition(factorization::SplitRefinedLU, x::AbstractVecOrMat)
    device_solve = factorization.device_solve
    if device_solve !== nothing
        solution = device_solve(ComplexF32.(x isa AbstractVector ? reshape(x, :, 1) : x))
        return x isa AbstractVector ? vec(solution) : solution
    end
    return ComplexF64.(factorization.factor \ ComplexF32.(x))
end

"""
    _split_precondition_double(factorization, x) -> ComplexF64 matrix

`(P L U)⁻¹ x` with the `ComplexF32` factors applied in `ComplexF64` arithmetic (blocked: each
factor panel is promoted once, so one application streams the factors once for all columns).
GMRES-IR needs the preconditioner at working precision to go beyond `κ eps(Float32) < 1`
(Carson & Higham 2018); the classical refinement keeps the `ComplexF32` solve.
"""
function _split_precondition_double(factorization::SplitRefinedLU, x::AbstractMatrix{ComplexF64})
    factor = factorization.factor
    factors = factor.factors
    n = size(factors, 1)
    y = x[factor.p, :]
    width = max(1, min(256, DENSE_SPLIT_PANEL_BYTES ÷ (16 * n)))
    for first_column in 1:width:n                                   # unit lower triangular, forward
        last_column = min(first_column + width - 1, n)
        block = first_column:last_column
        panel = ComplexF64.(view(factors, first_column:n, block))
        b = length(block)
        @views ldiv!(UnitLowerTriangular(panel[1:b, 1:b]), y[block, :])
        last_column < n && @views mul!(y[(last_column + 1):n, :], panel[(b + 1):end, :], y[block, :], -one(ComplexF64), one(ComplexF64))
    end
    for last_column in n:-width:1                                   # upper triangular, backward
        first_column = max(1, last_column - width + 1)
        block = first_column:last_column
        panel = ComplexF64.(view(factors, 1:last_column, block))
        @views ldiv!(UpperTriangular(panel[block, :]), y[block, :])
        first_column > 1 && @views mul!(y[1:(first_column - 1), :], panel[1:(first_column - 1), :], y[block, :], -one(ComplexF64), one(ComplexF64))
    end
    return y
end

function Base.:\(factorization::SplitRefinedLU, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    target = rhs isa AbstractVector ? reshape(ComplexF64.(rhs), :, 1) : ComplexF64.(rhs)
    solution = _split_solve_columns(factorization, target)
    return rhs isa AbstractVector ? vec(solution) : solution
end

function _split_solve_columns(factorization::SplitRefinedLU, target::Matrix{ComplexF64})
    residual = similar(target)          # holds the final residual on every successful exit
    solution = _split_solve_columns!(factorization, target, residual)
    factorization.output_estimates = nothing
    if _dense_error_estimate_enabled()
        apply = isnothing(factorization.fallback) ? (r -> _split_precondition(factorization, r)) :
                                                    (r -> factorization.fallback \ r)
        # one streaming pass over the matrix: A d for the direction check and |A| |x| (columns of x
        # appended) for the residual-rounding bound
        absolute_solution = Ref{Matrix{Float64}}()
        multiply = function (y)
            k = size(y, 2)
            combined = hcat(y, solution)
            absolute = Matrix{Float64}(undef, size(combined))
            product = _split_mul!(similar(combined), factorization.store, factorization.matrix, combined,
                                  one(ComplexF64), zero(ComplexF64); absolute)
            absolute_solution[] = absolute[:, (k + 1):end]
            return product[:, 1:k]
        end
        # M⁻ᴴ: not available for a factor held on the device, so the componentwise bound is unknown there
        apply_adjoint = !isnothing(factorization.fallback) ? (r -> factorization.fallback' \ r) :
                        isnothing(factorization.factor) ? nothing :
                        (r -> ComplexF64.(factorization.factor' \ ComplexF32.(r)))
        rounding = () -> _residual_rounding(target, absolute_solution[], size(solution, 1))
        factorization.estimated_error, factorization.estimate_direction_check, factorization.error_bound =
            _dense_error_estimate(apply, multiply, solution, residual, target; apply_adjoint, rounding)
    else
        factorization.estimated_error, factorization.estimate_direction_check = Float64[], Bool[]
        factorization.error_bound = zeros(0, 0)
    end
    return solution
end

function _split_solve_columns!(factorization::SplitRefinedLU, target::Matrix{ComplexF64}, residual)
    if isnothing(factorization.fallback)
        solution = _split_precondition(factorization, target)
        previous = Inf
        reason = nothing
        for iteration in 0:SPLIT_REFINEMENT_MAX_ITERATIONS
            ratios, normwise = _split_acceptance!(residual, factorization, target, solution)
            ratio = maximum(ratios; init=0.0)
            if !all(isfinite, normwise) || isnan(ratio)
                reason = "non-finite residual after $iteration refinement steps"
                break
            end
            if ratio <= 1
                factorization.backward_error = maximum(normwise; init=0.0)
                return solution
            end
            iteration == SPLIT_REFINEMENT_MAX_ITERATIONS && break
            if ratio > previous / 2
                reason = "refinement stalled at $(ratio)x the Float64 backward error after $iteration steps" *
                         _componentwise_note(factorization)
                break
            end
            previous = ratio
            solution .+= _split_precondition(factorization, residual)
            factorization.iterations = max(factorization.iterations, iteration + 1)
        end
        isnothing(reason) &&
            (reason = "refinement did not reach the Float64 backward error in $(SPLIT_REFINEMENT_MAX_ITERATIONS) steps" *
                      _componentwise_note(factorization))
        if _dense_gmres_ir_enabled() && factorization.device_solve === nothing
            accepted, gmres_reason = _split_gmres_ir!(factorization, target, solution, residual)
            accepted && return solution
            reason *= "; GMRES-IR: " * gmres_reason
        end
        _split_fall_back!(factorization, reason)
    end
    solution, normwise = _refine_fallback!(x -> _split_acceptance!(residual, factorization, target, x),
                                           factorization.fallback, factorization.fallback \ target, residual, factorization)
    factorization.backward_error = maximum(normwise; init=0.0)
    return solution
end

_componentwise_note(factorization) = isnan(factorization.componentwise_tolerance) ? "" :
    " (componentwise backward error $(factorization.componentwise_backward_error), tolerance $(factorization.componentwise_tolerance))"

"""
    _split_gmres_ir!(factorization, target, solution, residual) -> (accepted, reason)

Restarted, right-preconditioned GMRES on `M y = r` for all columns in lockstep (one streamed
product per Arnoldi step serves every column), `ComplexF64` Arnoldi, the `ComplexF32` LU factors
applied in `ComplexF64` as preconditioner (`_split_precondition_double`). After each cycle the true residual is recomputed and the same acceptance test (normwise and componentwise)
as the refinement decides. `solution` is updated in place.
"""
function _split_gmres_ir!(factorization::SplitRefinedLU, target, solution, residual)
    n, columns = size(target)
    m = DENSE_GMRES_IR_RESTART
    for cycle in 1:DENSE_GMRES_IR_MAX_CYCLES
        ratios, normwise = _split_acceptance!(residual, factorization, target, solution)
        # NaN does not compare greater than one: check before an empty active set
        # can be mistaken for convergence after a non-finite Float32 solve.
        (all(isfinite, normwise) && !any(isnan, ratios)) || return false, "non-finite residual"
        active = findall(>(1), ratios)
        isempty(active) && (factorization.backward_error = maximum(normwise; init=0.0); return true, "")
        k = length(active)
        basis = Array{ComplexF64}(undef, n, m + 1, k)
        hessenberg = zeros(ComplexF64, m + 1, m, k)
        beta = [norm(view(residual, :, column)) for column in active]
        for (local_column, column) in enumerate(active)
            @views basis[:, 1, local_column] .= residual[:, column] ./ beta[local_column]
        end
        preconditioned = Matrix{ComplexF64}(undef, n, k)
        product = Matrix{ComplexF64}(undef, n, k)
        steps = 0
        for j in 1:m
            preconditioned .= _split_precondition_double(factorization, basis[:, j, :])
            _split_mul!(product, factorization.store, factorization.matrix, preconditioned, one(ComplexF64), zero(ComplexF64))
            for local_column in 1:k
                w = view(product, :, local_column)
                for i in 1:j                                    # modified Gram-Schmidt
                    v = view(basis, :, i, local_column)
                    h = dot(v, w)
                    hessenberg[i, j, local_column] = h
                    w .-= h .* v
                end
                h = norm(w)
                hessenberg[j + 1, j, local_column] = h
                @views basis[:, j + 1, local_column] .= iszero(h) ? zero(ComplexF64) : w ./ h
            end
            steps = j
            factorization.gmres_iterations += 1
        end
        for (local_column, column) in enumerate(active)
            rhs_small = zeros(ComplexF64, steps + 1)
            rhs_small[1] = beta[local_column]
            y = view(hessenberg, 1:steps+1, 1:steps, local_column) \ rhs_small    # least squares (QR)
            correction = view(basis, :, 1:steps, local_column) * y
            @views solution[:, column] .+= vec(_split_precondition_double(factorization, reshape(correction, :, 1)))
        end
    end
    ratios, normwise = _split_acceptance!(residual, factorization, target, solution)
    maximum(ratios; init=0.0) <= 1 && (factorization.backward_error = maximum(normwise); return true, "")
    return false, "not converged after $(DENSE_GMRES_IR_MAX_CYCLES) restarts of $(DENSE_GMRES_IR_RESTART) " *
                  "(worst $(maximum(ratios))x the Float64 backward error)" * _componentwise_note(factorization)
end

# ---------------------------------------------------------------------------------------------
# Experimental: iterative solve of the condensed dense system without any LU factorization.
#
# `BLAB_COUPLED_DENSE_SOLVER=gmres` (default `lu`). Mixed-precision iterative refinement whose
# inner solver is restarted GMRES in ComplexF32 on the stored `hi` part (one ComplexF32 matrix
# product per Arnoldi step serves every right-hand side), outer residuals in ComplexF64 against the
# exact split matrix. No factor is ever allocated: the dense system needs 8 n^2 bytes instead of
# 16 n^2, and the work per right-hand side is O(n^2) per iteration instead of O(n^3) once.
# If the iteration does not reach the tolerance within its budget, the solve falls back to the
# usual SplitRefinedLU, so results never silently degrade.
#
#   BLAB_COUPLED_DENSE_GMRES_TOL      relative residual target per column (default 1e-10)
#   BLAB_COUPLED_DENSE_GMRES_RESTART  Arnoldi steps per restart (default 100)
#   BLAB_COUPLED_DENSE_GMRES_MAXITER  total inner steps per solve before falling back (default 2000)
#   BLAB_COUPLED_DENSE_GMRES_PRECOND  jacobi (default) | none
# ---------------------------------------------------------------------------------------------

_dense_solver_mode() = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_SOLVER", "lu")))
_dense_gmres_setting(name, default) = something(tryparse(typeof(default), get(ENV, name, string(default))), default)

mutable struct SplitGMRES
    matrix::SplitDenseMatrix
    store::Matrix{ComplexF32}
    preconditioner::Vector{ComplexF32}       # right Jacobi scaling (ones for :none)
    tolerance::Float64
    restart::Int
    max_inner::Int
    inner_iterations::Int                    # total Arnoldi steps over all solves
    outer_iterations::Int
    relative_residual::Float64               # worst column of the last solve (ComplexF64 residual)
    solve_seconds::Float64
    matvec_seconds::Float64
    fallback::Union{Nothing,SplitRefinedLU}
    fallback_reason::Union{Nothing,String}
    # per-output estimates of the last coupled solve without fallback: unknown (NaN), see
    # _store_output_estimates! (GMRES gives no componentwise bound)
    output_estimates::Union{Nothing,Dict{Symbol,Matrix{Float64}}}
end

function SplitGMRES(matrix::SplitDenseMatrix)
    all(isfinite, matrix.hi) || throw(ArgumentError("dense coupled matrix has non-finite entries"))
    n = size(matrix, 1)
    mode = lowercase(strip(get(ENV, "BLAB_COUPLED_DENSE_GMRES_PRECOND", "jacobi")))
    preconditioner = ones(ComplexF32, n)
    if mode == "jacobi"
        @inbounds for i in 1:n
            d = matrix.hi[i, i]
            preconditioner[i] = iszero(d) ? one(ComplexF32) : inv(d)
        end
    elseif mode != "none"
        error("BLAB_COUPLED_DENSE_GMRES_PRECOND must be jacobi or none; got $(repr(mode)).")
    end
    return SplitGMRES(matrix, matrix.hi, preconditioner,
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_TOL", 1e-10),
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_RESTART", 100),
                      _dense_gmres_setting("BLAB_COUPLED_DENSE_GMRES_MAXITER", 2000),
                      0, 0, NaN, 0.0, 0.0, nothing, nothing, nothing)
end

Base.size(solver::SplitGMRES, dims...) = size(solver.store, dims...)
release_split_factor!(solver::SplitGMRES) = (solver.fallback === nothing || release_split_factor!(solver.fallback); nothing)

function Base.:\(solver::SplitGMRES, rhs::AbstractVecOrMat)
    all(isfinite, rhs) || throw(ArgumentError("dense coupled right-hand side has non-finite entries"))
    solver.output_estimates = nothing
    solver.fallback === nothing || return solver.fallback \ rhs
    started = time_ns()
    target = rhs isa AbstractVector ? reshape(ComplexF64.(rhs), :, 1) : ComplexF64.(rhs)
    solution, ok, reason = _gmres_ir_solve(solver, target)
    solver.solve_seconds += (time_ns() - started) / 1.0e9
    if !ok
        solver.fallback_reason = "GMRES fell back to the Float32 LU with refinement: " * reason
        @warn solver.fallback_reason
        solver.fallback = SplitRefinedLU(solver.matrix)
        solution = solver.fallback \ target
    end
    return rhs isa AbstractVector ? vec(solution) : solution
end

function _gmres_ir_solve(solver::SplitGMRES, target::Matrix{ComplexF64})
    n, columns = size(target)
    solution = zeros(ComplexF64, n, columns)
    residual = similar(target)
    norms = [max(norm(view(target, :, j)), floatmin(Float64)) for j in 1:columns]
    budget = solver.max_inner
    previous = Inf
    for outer in 0:20
        copyto!(residual, target)
        _split_mul!(residual, solver.store, solver.matrix, solution, -one(ComplexF64), one(ComplexF64))
        relative = [norm(view(residual, :, j)) / norms[j] for j in 1:columns]
        worst = maximum(relative; init=0.0)
        solver.relative_residual = worst
        isfinite(worst) || return solution, false, "non-finite residual"
        worst <= solver.tolerance && return solution, true, ""
        budget <= 0 && return solution, false, "no convergence within $(solver.max_inner) inner steps (residual $(worst))"
        outer > 0 && worst > previous / 2 && return solution, false, "outer refinement stalled at $(worst)"
        previous = worst
        active = findall(>(solver.tolerance), relative)
        # inner tolerance: the remaining reduction, but no tighter than Float32 can deliver
        inner_tolerance = max(solver.tolerance / worst, 1e-4)
        correction, steps = try
            _gmres_f32(solver, ComplexF32.(residual[:, active]), inner_tolerance, budget)
        catch err
            err isa SingularException || rethrow()
            return solution, false, "Float32 GMRES correction has a singular projected system: $(sprint(showerror, err))"
        end
        budget -= steps
        solution[:, active] .+= correction
        solver.outer_iterations += 1
    end
    return solution, false, "outer refinement limit reached"
end

"""
Restarted, right-preconditioned GMRES in ComplexF32 for all columns in lockstep, Givens-rotated
least squares per column. Returns the ComplexF64 correction and the number of Arnoldi steps.
"""
function _gmres_f32(solver::SplitGMRES, rhs::Matrix{ComplexF32}, tolerance::Float64, budget::Int)
    n, k = size(rhs)
    m = solver.restart
    P = solver.preconditioner
    x = zeros(ComplexF32, n, k)
    r = copy(rhs)
    bnorm = [max(norm(view(rhs, :, j)), floatmin(Float32)) for j in 1:k]
    V = Array{ComplexF32}(undef, n, m + 1, k)
    H = zeros(ComplexF32, m + 1, m, k)
    cs = zeros(ComplexF32, m, k); sn = zeros(ComplexF32, m, k)
    g = zeros(ComplexF32, m + 1, k)
    Z = Matrix{ComplexF32}(undef, n, k)
    W = Matrix{ComplexF32}(undef, n, k)
    steps = 0
    previous = Inf
    while steps < budget
        beta = [norm(view(r, :, j)) for j in 1:k]
        current = maximum(beta[j] / bnorm[j] for j in 1:k)
        current <= tolerance && break
        # A restart that did not halve the residual means Float32 has reached its floor
        # (about cond * eps(Float32)); the ComplexF64 outer refinement continues from here.
        current > previous / 2 && break
        previous = current
        fill!(H, 0); fill!(g, 0)
        for j in 1:k
            @views V[:, 1, j] .= r[:, j] ./ max(beta[j], floatmin(Float32))
            g[1, j] = beta[j]
        end
        done = falses(k)
        used = 0
        column_steps = zeros(Int, k)
        for i in 1:m
            @views Z .= V[:, i, :] .* P                      # right preconditioning
            t = time_ns()
            mul!(W, solver.store, Z)
            solver.matvec_seconds += (time_ns() - t) / 1.0e9
            steps += 1; used = i
            for j in 1:k
                done[j] && continue
                w = view(W, :, j)
                for l in 1:i                                  # modified Gram-Schmidt
                    v = view(V, :, l, j)
                    h = dot(v, w)
                    H[l, i, j] = h
                    BLAS.axpy!(-h, v, w)
                end
                hn = norm(w)
                H[i + 1, i, j] = hn
                @views V[:, i + 1, j] .= iszero(hn) ? zero(ComplexF32) : w ./ hn
                for l in 1:(i - 1)                            # apply previous rotations
                    a = H[l, i, j]; b = H[l + 1, i, j]
                    H[l, i, j] = conj(cs[l, j]) * a + conj(sn[l, j]) * b
                    H[l + 1, i, j] = -sn[l, j] * a + cs[l, j] * b
                end
                a = H[i, i, j]; b = H[i + 1, i, j]
                denom = sqrt(abs2(a) + abs2(b))
                c, s = iszero(denom) ? (one(ComplexF32), zero(ComplexF32)) : (a / denom, b / denom)
                cs[i, j] = c; sn[i, j] = s
                H[i, i, j] = conj(c) * a + conj(s) * b
                H[i + 1, i, j] = 0
                g[i + 1, j] = -s * g[i, j]
                g[i, j] = conj(c) * g[i, j]
                column_steps[j] = i
                abs(g[i + 1, j]) / bnorm[j] <= tolerance && (done[j] = true)
            end
            (all(done) || steps >= budget) && break
        end
        for j in 1:k                                          # update x_j with its least squares
            steps_j = column_steps[j]
            steps_j == 0 && continue
            y = UpperTriangular(view(H, 1:steps_j, 1:steps_j, j)) \ view(g, 1:steps_j, j)
            @views x[:, j] .+= (V[:, 1:steps_j, j] * y) .* P
        end
        mul!(r, solver.store, x)                             # true ComplexF32 residual for the restart
        r .= rhs .- r
    end
    solver.inner_iterations += steps
    return ComplexF64.(x), steps
end
