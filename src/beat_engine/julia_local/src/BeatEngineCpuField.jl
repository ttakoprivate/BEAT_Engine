function evaluate_galerkin_field_cpu(
    eval_points,
    mesh::BoundaryMesh{T},
    pressure,
    q_neumann,
    k::T,
    cache::FieldEvaluationCache{T};
    kernel::Symbol=:scalar,
) where {T<:AbstractFloat}
    k = outgoing_wavenumber(k)
    kernel === :simd && return _beat_cpu_field_simd(eval_points, pressure, q_neumann, k, cache)
    kernel === :scalar || error("CPU field kernel must be :simd or :scalar; got $(repr(kernel)).")
    point_count = length(eval_points)
    point_count == 0 && return Complex{T}[]

    potentials = Vector{Complex{T}}(undef, point_count)
    source_count = length(cache.source_points)

    Threads.@threads for point_index in 1:point_count
        x = SVector{3,T}(T(eval_points[point_index][1]), T(eval_points[point_index][2]), T(eval_points[point_index][3]))
        potential = zero(Complex{T})
        for source_index in 1:source_count
            y = cache.source_points[source_index]
            r_vec = y - x
            radius = norm(r_vec)
            radius == zero(T) && continue

            phase = k * radius
            green_scale = inv(T(4.0) * T(pi) * radius)
            green = Complex{T}(cos(phase) * green_scale, sin(phase) * green_scale)
            normal_projection = dot(r_vec, cache.source_normals[source_index]) / radius
            double_layer = green * (Complex{T}(0, k) - inv(radius)) * normal_projection

            face = cache.source_faces[source_index]
            basis = cache.basis_values[source_index]
            weight = cache.source_weights[source_index]
            p_source = (
                basis[1] * pressure[face[1]] +
                basis[2] * pressure[face[2]] +
                basis[3] * pressure[face[3]]
            ) * weight
            q_source = q_neumann[cache.source_elements[source_index]] * weight
            potential += double_layer * p_source - green * q_source
        end
        potentials[point_index] = potential
    end

    return potentials
end

"""
    galerkin_field_error_bound_cpu(eval_points, pressure_bound, neumann_bound, k, cache) -> Vector{Float64}

Propagate componentwise absolute bounds of the boundary data through the field representation of
`evaluate_galerkin_field_cpu` (step 2 of docs/ROADMAP.md, display only). With `|Δp_a| <=
pressure_bound[a]` (nodes) and `|Δq_e| <= neumann_bound[e]` (elements) the field error at `x` is at
most `Σ_s |w_s| (|∂G/∂n| Σ_a |φ_a| pressure_bound[a] + |G| neumann_bound[e_s])`, the same sum over
the same quadrature sources as the field, at the evaluation point rounded to the cache precision as
the field kernel does. Kernel moduli are taken in Float64. `NaN` bounds give `NaN`.
Covers the propagated input error only; the rounding of the field sum itself is not included.
"""
function galerkin_field_error_bound_cpu(eval_points, pressure_bound::AbstractVector, neumann_bound::AbstractVector,
                                        k, cache::FieldEvaluationCache)
    k = ComplexF64(outgoing_wavenumber(k))
    bounds = zeros(Float64, length(eval_points))
    source_count = length(cache.source_points)
    # the field kernel rounds the point to the cache precision first (REVIEW-22-9abe18c P2): use that point
    R = eltype(cache.source_weights)
    Threads.@threads for point_index in eachindex(eval_points)
        x = SVector{3,Float64}(R(eval_points[point_index][1]), R(eval_points[point_index][2]), R(eval_points[point_index][3]))
        total = 0.0
        for source_index in 1:source_count
            r_vec = SVector{3,Float64}(cache.source_points[source_index]) - x
            radius = norm(r_vec)
            radius == 0.0 && continue
            green = abs(exp(im * k * radius)) / (4pi * radius)
            normal = SVector{3,Float64}(cache.source_normals[source_index])
            # computed projection plus its absolute Float64 rounding (cancellation, REVIEW-22-2fd01ac P2)
            projection = (abs(dot(r_vec, normal)) + _projection_rounding(r_vec, normal)) / radius
            double_layer = green * abs(im * k - inv(radius)) * projection
            face = cache.source_faces[source_index]
            basis = cache.basis_values[source_index]
            p_bound = abs(Float64(basis[1])) * pressure_bound[face[1]] +
                      abs(Float64(basis[2])) * pressure_bound[face[2]] +
                      abs(Float64(basis[3])) * pressure_bound[face[3]]
            q_bound = neumann_bound[cache.source_elements[source_index]]
            total += abs(Float64(cache.source_weights[source_index])) * (double_layer * p_bound + green * q_bound)
        end
        bounds[point_index] = total
    end
    return bounds
end

# Absolute Float64 rounding bound of `dot(r_vec, normal)` with `r_vec = y − x` itself rounded
# (relative u per component) and the 3-term sum (γ_3): (u + γ_3(1 + u)) Σ|r_i n_i| <= 8u Σ|r_i n_i|,
# generous also for the second-order terms of the division by the rounded radius.
_projection_rounding(r_vec, normal) = 8 * (eps(Float64) / 2) * sum(abs.(r_vec .* normal))

"""
    galerkin_field_float64_reference_cpu(eval_points, pressure, neumann, k, cache) -> (values, bound)

Float64 evaluation of the field representation of `evaluate_galerkin_field_cpu` from the given
boundary data (step 2 of docs/ROADMAP.md, display only), with a bound of its own Float64 rounding.
Same quadrature sources as `cache` (its stored geometry, converted exactly to Float64), the
wavenumber `k` as given and the evaluation points as given (not rounded to the cache precision).
`|field_T − values| + bound` therefore covers every rounding of a field evaluation in the cache
precision `T` from the same stored data: the field sum, the kernel (phase `k r`, `1/r`, sincos
approximation of the SIMD kernel), the products with the data and the rounding of the points.
Not covered: the rounding of `k` and of the geometry to `T` (the operator itself, step 4).

`bound` is a generous first-order bound of the Float64 rounding: per source term
`(32 + 4 |k| r) u` times its modulus (kernel, phase `k r` with `r` rounded, products), plus the
absolute rounding `8u Σ|r_i n_i| / r` of the normal projection times the rest of the double-layer
modulus and the pressure modulus (not relative to the computed projection, which can cancel for
nearly tangential sources), plus `γ_S` times the sum of the term moduli for the sequential sum over
`S` sources (u = eps(Float64)/2).
"""
function galerkin_field_float64_reference_cpu(eval_points, pressure::AbstractVector, neumann::AbstractVector,
                                              k, cache::FieldEvaluationCache)
    k = ComplexF64(outgoing_wavenumber(k))
    point_count = length(eval_points)
    values = Vector{ComplexF64}(undef, point_count)
    bounds = zeros(Float64, point_count)
    source_count = length(cache.source_points)
    u = eps(Float64) / 2
    gamma = source_count * u / (1 - source_count * u)
    Threads.@threads for point_index in 1:point_count
        x = SVector{3,Float64}(Float64(eval_points[point_index][1]), Float64(eval_points[point_index][2]),
                               Float64(eval_points[point_index][3]))
        potential = zero(ComplexF64)
        magnitude = 0.0
        term_error = 0.0
        for source_index in 1:source_count
            r_vec = SVector{3,Float64}(cache.source_points[source_index]) - x
            radius = norm(r_vec)
            radius == 0.0 && continue
            green = exp(im * k * radius) / (4pi * radius)
            normal = SVector{3,Float64}(cache.source_normals[source_index])
            projection = dot(r_vec, normal) / radius
            double_layer = green * (im * k - inv(radius)) * projection
            # absolute rounding of the projection: under cancellation in dot(r, n) it is not relative
            # to the computed projection (REVIEW-22-2fd01ac P2), so it is propagated separately
            projection_error = _projection_rounding(r_vec, normal) / radius
            face = cache.source_faces[source_index]
            basis = cache.basis_values[source_index]
            weight = Float64(cache.source_weights[source_index])
            p_source = (Float64(basis[1]) * ComplexF64(pressure[face[1]]) +
                        Float64(basis[2]) * ComplexF64(pressure[face[2]]) +
                        Float64(basis[3]) * ComplexF64(pressure[face[3]])) * weight
            q_source = ComplexF64(neumann[cache.source_elements[source_index]]) * weight
            potential += double_layer * p_source - green * q_source
            p_modulus = (abs(Float64(basis[1])) * abs(ComplexF64(pressure[face[1]])) +
                         abs(Float64(basis[2])) * abs(ComplexF64(pressure[face[2]])) +
                         abs(Float64(basis[3])) * abs(ComplexF64(pressure[face[3]]))) * abs(weight)
            term = abs(double_layer) * p_modulus + abs(green) * abs(q_source)
            magnitude += term
            term_error += (32 + 4 * abs(k) * radius) * u * term +
                          (1 + 8u) * abs(green) * abs(im * k - inv(radius)) * projection_error * p_modulus
        end
        values[point_index] = potential
        bounds[point_index] = (1 + 4u) * (term_error + gamma * magnitude)
    end
    return values, bounds
end
