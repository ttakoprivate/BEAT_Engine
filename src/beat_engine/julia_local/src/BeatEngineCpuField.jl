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
the same quadrature sources as the field. Kernel moduli are taken in Float64. `NaN` bounds give `NaN`.
Covers the propagated input error only; the rounding of the field sum itself is not included.
"""
function galerkin_field_error_bound_cpu(eval_points, pressure_bound::AbstractVector, neumann_bound::AbstractVector,
                                        k, cache::FieldEvaluationCache)
    k = ComplexF64(outgoing_wavenumber(k))
    bounds = zeros(Float64, length(eval_points))
    source_count = length(cache.source_points)
    Threads.@threads for point_index in eachindex(eval_points)
        x = SVector{3,Float64}(eval_points[point_index][1], eval_points[point_index][2], eval_points[point_index][3])
        total = 0.0
        for source_index in 1:source_count
            r_vec = SVector{3,Float64}(cache.source_points[source_index]) - x
            radius = norm(r_vec)
            radius == 0.0 && continue
            green = abs(exp(im * k * radius)) / (4pi * radius)
            projection = abs(dot(r_vec, SVector{3,Float64}(cache.source_normals[source_index]))) / radius
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
