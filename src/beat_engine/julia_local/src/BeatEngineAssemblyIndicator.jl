module BeatEngineAssemblyIndicator

using LinearAlgebra, Random, SparseArrays
using ..BeatEngineCore
using ..BeatEngineCoupledCondensed
const CoreBEM = BeatEngineCore
const Condensed = BeatEngineCoupledCondensed

export sample_equations, sample_bem_rows, build_indicator_reference, capture_coupled_state,
    assembly_indicator, physical_fields

const physical_fields = (:fem_pressure, :bem_pressure, :interface_flux, :diaphragm_velocity, :voice_coil_current)

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
        isempty(stratum) && iszero(count) && continue
        1 <= count <= length(stratum) || throw(ArgumentError("Every nonempty stratum requires a positive sample."))
        selected = count == length(stratum) ? collect(stratum) : sort(collect(stratum)[randperm(rng, length(stratum))[1:count]])
        push!(groups, length(rows)+1:length(rows)+count)
        append!(rows, selected)
        append!(probability, fill(count / length(stratum), count))
    end
    return (; rows, probability, groups)
end

# Horvitz-Thompson projection for a fixed output. The standard error uses the
# finite population correction in each stratum and the magnitude of complex
# deviations. Selecting the displayed peak from the same draw can add bias.
function project_output(sample, dual, residual, direct)
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

function output_gradient(s, field, i)
    g = zeros(ComplexF64, size(s.factorization, 1))
    ranges = (bem_pressure=s.bem_range, diaphragm_velocity=s.mechanical_range, voice_coil_current=s.electrical_range)
    if haskey(ranges, field)
        g[getproperty(ranges, field)[i]] = 1
        return g
    end
    e = s.interface_elimination_data
    pressure = zeros(ComplexF64, length(e.bem_of_gamma))
    if field === :interface_flux
        if hasproperty(e, :mass_operator)
            for (block, schur) in zip(e.mass_operator.blocks, e.schur_blocks)
                j = findfirst(==(i), block.dofs)
                isnothing(j) || (pressure[block.rows] .= conj.(schur[j, :]))
            end
        else
            pressure .= conj.(e.schur_solution[i, :])
        end
        isempty(s.mechanical_range) || (g[s.mechanical_range] .= conj.(e.motion_solution[i, :]))
    elseif field === :fem_pressure
        c = s.condensation
        retained = findfirst(==(i), c.retained_vertices)
        if isnothing(retained)
            interior = findfirst(==(i), c.interior_vertices)
            basis = zeros(ComplexF64, c.interior_count)
            basis[interior] = 1
            dual = c.factorization' \ basis
            pressure .= -(c.interior_retained' * dual)
            if c.transducer_condensed && !isempty(s.mechanical_range)
                g[s.mechanical_range] .= conj(CoreBEM.neumann_scale(s.density, s.omega)) .* (c.motion_interior' * dual)
            end
        else
            pressure[retained] = 1
        end
    else
        throw(ArgumentError("Unsupported physical output $field."))
    end
    for (j, vertex) in enumerate(e.bem_of_gamma)
        g[s.bem_range[vertex]] += pressure[j]
    end
    return g
end

"""
    assembly_indicator(mixed, reference, excitation, state, sample; output_indices=nothing)

CPU prototype for flux-eliminated, in-memory split LU. The caller builds an
independent Float64 reference from the original inputs, passing identity BEM A
and zero C to the normal builder. Its FEM, interface, loads and output recovery
remain physical. Only the sampled BEM rows are then evaluated with quadrature.

The scalar per field is a first-order estimate of the largest pointwise complex
relative change for this draw. It is not a bound. Reuse one forward solve to
locate peaks across all field components, then evaluate those linear outputs
with adjoint solves using the same existing factor. The signed assembly
residual excludes the mixed algebraic residual. Supply one fixed index per
field to evaluate preselected outputs on an independent draw.
"""
function assembly_indicator(mixed, reference, excitation, state, sample; output_indices=nothing)
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
    rows, pi = sample.rows, sample.probability
    length(rows) == length(pi) && allunique(rows) && all(p -> 0 < p <= 1, pi) ||
        throw(ArgumentError("Invalid sampled rows or inclusion probabilities."))
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
    moved = if isnothing(output_indices)
        weighted = zeros(ComplexF64, length(state.x))
        weighted[rows] = assembly_residual ./ pi
        correction = ComplexF64.(f.factor \ ComplexF32.(weighted))
        replay(reference, excitation, state.x + correction).fields
    end
    fields = Dict{Symbol,NamedTuple}()
    for field in physical_fields
        old = getproperty(state.fields, field)
        isempty(old) && continue
        index = if isnothing(output_indices)
            predicted = getproperty(moved, field)
            argmax(relative_change.(predicted .- old, predicted))
        else
            output_indices[field]
        end
        g = output_gradient(reference, field, index)
        adjoint_state = ComplexF64.(f.factor' \ ComplexF32.(g))
        direct = getproperty(ref_at_x.fields, field)[index] - old[index]
        delta, stderr = project_output(sample, adjoint_state, assembly_residual, direct)
        fields[field] = (; index, delta, relative=relative_change(delta, old[index] + delta),
                         absolute=abs(delta), direct, sampling_stderr=stderr,
                         inconclusive=iszero(old[index] + delta) || abs(delta) <= stderr)
    end
    return (; fields, assembly_residual, mixed_residual, rows, probability=pi)
end

end
