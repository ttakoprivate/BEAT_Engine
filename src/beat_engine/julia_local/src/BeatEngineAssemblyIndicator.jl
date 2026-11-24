module BeatEngineAssemblyIndicator

using LinearAlgebra, Random, SparseArrays
using ..BeatEngineCore
using ..BeatEngineCoupledCondensed
const CoreBEM = BeatEngineCore
const Condensed = BeatEngineCoupledCondensed

export sample_equations, sample_bem_rows, build_indicator_reference, capture_coupled_state,
    assembly_indicator, physical_fields

const physical_fields = (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)

"""Build the independent FEM/recovery model without performing dense BEM quadrature."""
function build_indicator_reference(fem, bem, interface, frequency::Float64, speed::Float64, density::Float64; kwargs...)
    n = length(bem.vertices)
    operators = (bem_lhs=Diagonal(ones(ComplexF64, n)),
                 bem_rhs_operator=spzeros(ComplexF64, n, length(bem.faces)),
                 bem_interface_block=spzeros(ComplexF64, n, length(interface.fem_vertex_indices)),
                 wavenumber=(2pi * frequency) / speed)
    return build_condensed_coupled_system(fem, bem, interface, frequency, speed, density;
        bem_operators=() -> operators, kwargs...)
end

"""Sample each disjoint stratum without replacement. Census strata retain weight one."""
function sample_equations(rng, n, strata, counts)
    length(strata) == length(counts) || throw(DimensionMismatch("One budget is required per stratum."))
    sort(vcat(collect.(strata)...)) == collect(1:n) || throw(ArgumentError("Strata must partition all equations."))
    rows = Int[]
    probability = Float64[]
    groups = UnitRange{Int}[]
    for (stratum, count) in zip(strata, counts)
        count isa Integer && !(count isa Bool) || throw(ArgumentError("Sample budgets must be integers, not Bool."))
        isempty(stratum) && iszero(count) && continue
        1 <= count <= length(stratum) || throw(ArgumentError("Every nonempty stratum requires a positive sample."))
        selected = count == length(stratum) ? collect(stratum) : sort(collect(stratum)[randperm(rng, length(stratum))[1:count]])
        push!(groups, length(rows)+1:length(rows)+count)
        append!(rows, selected)
        append!(probability, fill(count / length(stratum), count))
    end
    return (; rows, probability, groups)
end

# Explicit calibration samples must have the same stratified design as generated samples.
function validate_sample(sample, n)
    rows, probability, groups = sample.rows, sample.probability, sample.groups
    n > 0 && isempty(rows) && throw(ArgumentError("A nonempty system requires sampled rows."))
    length(rows) == length(probability) || throw(DimensionMismatch("One probability is required per sampled row."))
    all(row -> row isa Integer && !(row isa Bool) && 1 <= row <= n, rows) && allunique(rows) ||
        throw(ArgumentError("Sampled rows must be distinct equation indices."))
    all(p -> p isa Real && !(p isa Bool) && isfinite(p) && 0 < p <= 1, probability) ||
        throw(ArgumentError("Inclusion probabilities must be finite and in (0, 1]."))
    covered = Int[]
    for group in groups
        !isempty(group) && all(i -> i isa Integer && !(i isa Bool) && 1 <= i <= length(rows), group) ||
            throw(ArgumentError("Sample groups must contain valid nonempty position sets."))
        append!(covered, group)
        p = probability[first(group)]
        all(i -> probability[i] == p, group) ||
            throw(ArgumentError("Inclusion probability must be constant within each sample group."))
    end
    sort(covered) == collect(eachindex(rows)) ||
        throw(ArgumentError("Sample groups must partition all sampled positions."))
    return nothing
end

# Horvitz-Thompson projection for a fixed output. The standard error uses the
# finite population correction in each stratum and the magnitude of complex
# deviations. Selecting the displayed peak from the same draw can add bias.
function project_output(sample, dual, residual, direct)
    length(residual) == length(sample.rows) || throw(DimensionMismatch("One residual is required per sampled row."))
    validate_sample(sample, length(dual))
    contributions = conj.(dual[sample.rows]) .* residual
    delta = direct + sum(contributions ./ sample.probability)
    variance = 0.0
    for group in sample.groups
        p = sample.probability[first(group)]
        isone(p) && continue
        m = length(group)
        m == 1 && return (; delta, stderr=Inf)
        terms = view(contributions, group)
        mean_term = sum(terms) / m
        variance += m * (1 - p) / (p^2 * (m-1)) * sum(t -> abs2(t - mean_term), terms)
    end
    return (; delta, stderr=sqrt(variance))
end

# Existing quadrature kernels scatter by global DOF. Keep only the selected test
# rows while retaining every trial column, including singular and image pairs.
struct RowScatter
    values::Matrix{ComplexF64}
    slots::Vector{Int}
end

@inline function Base.getindex(a::RowScatter, i::Int, j::Int)
    slot = a.slots[i]
    return iszero(slot) ? 0.0im : a.values[slot, j]
end

@inline function Base.setindex!(a::RowScatter, v, i::Int, j::Int)
    slot = a.slots[i]
    iszero(slot) || (a.values[slot, j] = v)
    return v
end

"""Independently assemble complete A and C rows from a Float64 model's CPU quadrature cache."""
function sample_bem_rows(s, rows)
    s.scalar_type === Float64 && s.bem_backend === :cpu || throw(ArgumentError("Requires a Float64 CPU model."))
    bundle = s.cache.quadrature_bundles[s.regular_quadrature_order]
    data = bundle.cpu_assembly_cache
    singular = s.cache.base.singular_cache
    n = length(s.bem_mesh.vertices)
    m = length(s.bem_mesh.faces)
    allunique(rows) || throw(ArgumentError("BEM rows must be distinct."))
    slots = zeros(Int, n)
    slots[rows] .= eachindex(rows)
    single, double, adjoint_double, hyper = (RowScatter(zeros(ComplexF64, length(rows), width), slots)
                                             for width in (m, n, m, n))
    k = CoreBEM.outgoing_wavenumber(s.wavenumber)
    for test in data.indices
        any(i -> !iszero(slots[i]), data.elements[test].p1_dofs) || continue
        Condensed._condensed_accumulate_fused_test!(single, double, adjoint_double, hyper,
            data.elements, data.image_elements, test, data.indices, k, data.regular_quadrature, data.image_quadrature)
        CoreBEM._beat_cpu_accumulate_singular_test!(single, double, adjoint_double, hyper,
            data.elements, singular.pairs_by_test[test], singular.rules, k)
        for (image, elements, quadrature) in zip(data.image_singular_caches, data.image_elements, data.image_quadrature)
            CoreBEM._beat_cpu_accumulate_image_singular_delta_test!(single, double, adjoint_double, hyper,
                data.elements, elements, image.pairs_by_test[test], image.rules, k, data.regular_quadrature, quadrature)
        end
    end
    weights = CoreBEM.p1_symmetry_orbit_weights(s.bem_mesh, s.symmetry_mode)[rows]
    for scatter in (single, double, adjoint_double, hyper)
        scatter.values .*= weights
    end
    ops = (single_layer=single.values, double_layer=double.values,
           adjoint_double_layer=adjoint_double.values, hypersingular=hyper.values)
    a, negative_c = CoreBEM.burton_miller_neumann_matrices(ops,
        bundle.identity_p1_p1[rows, :], bundle.identity_p1_dp0[rows, :], s.wavenumber)
    return a, -negative_c
end

# Replay the ordinary RHS and physical reconstruction at a specified reduced state.
struct ProbeSolve{F}
    n::Int
    solve::F
end

Base.size(f::ProbeSolve, dim::Int) = dim in (1, 2) ? f.n : 1
Base.:\(f::ProbeSolve, b::AbstractMatrix) = f.solve(b)

function capture_coupled_state(f, s, excitation)
    rhs = Ref{Matrix{ComplexF64}}()
    state = Ref{Matrix{ComplexF64}}()
    intercept = ProbeSolve(size(s.factorization, 1), b -> begin
        rhs[] = ComplexF64.(b)
        state[] = ComplexF64.(f(b))
    end)
    fields = only(solve_condensed_coupled_excitations(merge(s, (factorization=intercept,)), [excitation]))
    return (; rhs=vec(rhs[]), x=vec(state[]), fields)
end

function replay(s, excitation, x)
    return capture_coupled_state(b -> reshape(x, :, 1), s, excitation)
end

# Outside the BEM equations a zero-BEM reference has the true coupled rows,
# including mechanical feedback. Pivoted LU stores L*U = A[p,:].
function factored_rows_product(factor::LU, rows, x)
    isempty(rows) && return ComplexF64[]
    z = UpperTriangular(factor.factors) * x
    y = UnitLowerTriangular(factor.factors) * z
    return y[invperm(factor.p)[rows]]
end

function relative_change(delta, ref)
    isfinite(delta) && isfinite(ref) || throw(DomainError((delta, ref), "Nonfinite physical output."))
    iszero(ref) && return iszero(delta) ? 0.0 : Inf
    return abs(delta) / abs(ref)
end

# Stratify acoustic equations by their largest normalized adjoint weight across
# the requested driver outputs. Half the budget samples the highest fifth. Every
# BEM row has positive inclusion probability; the driver equations are a census.
"""Stratify driver diagnostics; BEM budgets 2 or 3 can leave singleton strata (infinite standard error).
Use at least 4 BEM rows to avoid that budget-induced limitation when strata are large enough.
"""
function driver_sample(rng, system, state, dual, budget)
    bem = collect(system.bem_range)
    budget >= 1 || throw(ArgumentError("The BEM row budget must be positive."))
    count = min(budget, length(bem))
    n = length(state.x)
    mechanical, electrical = system.mechanical_range, system.electrical_range
    if count <= 1 || count == length(bem)
        return sample_equations(rng, n, (bem, mechanical, electrical),
                                (count, length(mechanical), length(electrical)))
    end
    scale = abs.(state.rhs[bem])
    matrix = system.factorization.matrix
    for j in eachindex(state.x)
        magnitude = abs(state.x[j])
        for (i, row) in enumerate(bem)
            scale[i] += abs(matrix[row, j]) * magnitude
        end
    end
    score = zeros(length(bem))
    for z in eachcol(dual)
        w = abs.(z[bem]) .* scale
        peak = maximum(w)
        iszero(peak) || (score .= max.(score, w ./ peak))
    end
    high_count = cld(length(bem), 5)
    order = sortperm(score; rev=true)
    high, rest = bem[order[1:high_count]], bem[order[high_count+1:end]]
    take = clamp(cld(count, 2), max(1, count-length(rest)), min(length(high), count-1))
    return sample_equations(rng, n, (high, rest, mechanical, electrical),
                            (take, count-take, length(mechanical), length(electrical)))
end

"""
    assembly_indicator(mixed, reference, excitation, state, sample=nothing; bem_budget=32, rng, output_indices)

Estimate complex assembly errors in diaphragm velocity and voice-coil current for
one excitation. Use the existing split factor for the adjoints and independently
assemble the selected Float64 equations, including the RHS. The returned
per-driver corrections and sampling standard errors are diagnostic estimates.
Pressure and flux require an independent dense check at the control frequencies.
An explicit probability sample supports calibration against all equations.
"""
function assembly_indicator(mixed, reference, excitation, state, sample=nothing;
                            bem_budget=32, rng=Random.default_rng(), output_indices=nothing)
    mixed.interface_elimination === reference.interface_elimination === :flux ||
        throw(ArgumentError("The prototype requires matching flux-eliminated systems."))
    f = mixed.factorization
    f isa Condensed.SplitRefinedLU && isnothing(f.spill_path) && !isnothing(f.factor) &&
        isnothing(f.fallback) || throw(ArgumentError("Requires an active, in-memory CPU split LU."))
    reference.scalar_type === Float64 && reference.factorization isa LU{ComplexF64} ||
        throw(ArgumentError("An independent Float64 CPU reference is required."))
    reference.condensation.backend === :mumps_seq && throw(ArgumentError("UMFPACK FEM recovery is required."))
    for key in (:bem_range, :mechanical_range, :electrical_range, :gamma_fem_vertices, :transducer_condensation,
                :regular_quadrature_order, :symmetry_mode)
        getproperty(mixed, key) == getproperty(reference, key) || throw(ArgumentError("Different reduced layouts."))
    end
    mixed.cache.singular_order == reference.cache.singular_order &&
        mixed.interface_elimination_data.bem_of_gamma == reference.interface_elimination_data.bem_of_gamma ||
        throw(ArgumentError("Different interface maps or singular quadrature."))
    for key in (:omega, :wavenumber, :density)
        a, b = getproperty(mixed, key), getproperty(reference, key)
        isfinite(a) && isfinite(b) && isapprox(a, b; rtol=8eps(Float32)) ||
            throw(ArgumentError("Mismatched model input $key."))
    end
    ranges = (; diaphragm_velocity=mixed.mechanical_range, voice_coil_current=mixed.electrical_range)
    if !isnothing(output_indices)
        all(k -> haskey(ranges, k), keys(output_indices)) ||
            throw(ArgumentError("Row estimates are available for driver current and velocity only."))
    end
    targets = Tuple{Symbol,Int,Int}[]
    for (field, range) in pairs(ranges)
        isempty(range) && continue
        !isnothing(output_indices) && !haskey(output_indices, field) && continue
        indices = isnothing(output_indices) ? eachindex(range) : (output_indices[field],)
        for i in indices
            push!(targets, (field, i, range[i]))
        end
    end
    isempty(targets) && throw(ArgumentError("The system has no driver outputs to estimate."))
    goals = zeros(ComplexF32, length(state.x), length(targets))
    for (j, (_, _, row)) in enumerate(targets)
        goals[row, j] = 1
    end
    dual = ComplexF64.(f.factor' \ goals)
    if isnothing(sample)
        sample = driver_sample(rng, mixed, state, dual, bem_budget)
    end
    validate_sample(sample, length(state.x))
    rows, pi = sample.rows, sample.probability
    isbem = in.(rows, Ref(reference.bem_range))
    bem = rows[isbem]
    ref_at_x = replay(reference, excitation, state.x)
    a, c = sample_bem_rows(reference, bem)
    defect = zeros(ComplexF64, length(rows))
    defect[isbem] = -(a * ref_at_x.fields.bem_pressure + c * ref_at_x.fields.bem_neumann)
    other = rows[.!isbem]
    defect[.!isbem] = ref_at_x.rhs[other] - factored_rows_product(reference.factorization, other, state.x)
    mixed_residual = state.rhs[rows] - f.matrix[rows, :] * state.x
    assembly_residual = defect - mixed_residual
    fields = Dict{Symbol,NamedTuple}()
    per_driver = Dict(field => NamedTuple[] for field in keys(ranges))
    for (j, (field, index, _)) in enumerate(targets)
        old = getproperty(state.fields, field)[index]
        direct = getproperty(ref_at_x.fields, field)[index] - old
        delta, stderr = project_output(sample, view(dual, :, j), assembly_residual, direct)
        predicted = old + delta
        item = (; index, delta, relative=relative_change(delta, predicted), absolute=abs(delta), direct,
                sampling_stderr=stderr, inconclusive=iszero(predicted) || abs(delta) <= stderr)
        push!(per_driver[field], item)
        if !haskey(fields, field) || item.relative > fields[field].relative
            fields[field] = item
        end
    end
    return (; fields, per_driver, assembly_residual, mixed_residual, rows, probability=pi)
end

end
