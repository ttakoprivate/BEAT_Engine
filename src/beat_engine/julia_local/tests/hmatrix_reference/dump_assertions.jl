# Loaded after the actual checker, in its process, with its decoded binary dump.
using Test
reference = hmat_load(ENV["HMAT_TRUTH_DIR"])
@test check_truth(reference, d) == "verified Float64 dump"
@test_throws ErrorException check_truth(merge(reference, (; model_scalar_type=:Float32)), d)
@test_throws ErrorException check_truth(merge(reference, (; dense_solver=:lu_float32_refined)), d)
@test_throws ErrorException check_truth(merge(reference, (; bem_backend=:metal)), d)
@test_throws ErrorException check_truth(merge(reference, (; assembly_eltype=:ComplexF32)), d)
@test_throws ErrorException check_truth(merge(reference, (; solver_scalar_type=:Float32)), d)
@test_throws ErrorException check_truth(merge(reference, (; solution_eltype=:ComplexF32)), d)
for k in (NaN, Inf, -Inf, 0., -1., 2pi*2000/330, 2pi*2000/342)
    @test_throws ErrorException check_truth(merge(reference, (; wavenumber=k)), d)
    @test_throws ErrorException check_truth(reference, merge(d, (; wavenumber=k)))
end
@test check_truth(reference, merge(d, (; wavenumber=Float64(Float32(d.wavenumber))))) == "verified Float64 dump"
@test_throws ErrorException check_truth(merge(reference, (; frequency_hz=2001.)), d)
mesh = reference.bem_mesh
vertices = copy(mesh.vertices)
vertices[end] += SVector(0.001,0.,0.)
@test_throws ErrorException check_truth(merge(reference, (; bem_mesh=BoundaryMesh(vertices, mesh.faces, mesh.physical_tags))), d)
tags = copy(mesh.physical_tags)
tags[end] += 1
@test_throws ErrorException check_truth(merge(reference, (; bem_mesh=BoundaryMesh(mesh.vertices,mesh.faces,tags))), d)
faces = copy(mesh.faces)
faces[1] = reverse(faces[1])
@test_throws ErrorException check_truth(merge(reference, (; bem_mesh=BoundaryMesh(mesh.vertices,faces,mesh.physical_tags))), d)
@test_throws ErrorException check_truth(merge(reference, (; rest=2 * reference.rest)), d)
@test_throws ErrorException check_truth(merge(reference, (; bem_flux=vcat(reference.bem_flux, spzeros(1,1)))), d)
b = copy(reference.rhs)
b[:,2] .*= -1
@test_throws ErrorException check_truth(merge(reference, (; rhs=b)), d)
b[:,2] .= 0
@test_throws ErrorException check_truth(merge(reference, (; rhs=b)), d)
for change in ((; regular_quadrature_order=1), (; singular_order=1), (; symmetry_mode=:xy))
    other = merge(reference, change)
    @test_throws ErrorException check_truth(other, d)
    withenv("HMAT_TRUTH_DISCRETISATION_STUDY"=>"1") do
        @test startswith(check_truth(other, d), "Float64 discretisation-study reference")
        @test_throws ErrorException check_truth(merge(other, (; wavenumber=Inf)), d)
    end
    withenv("HMAT_TRUTH_UNCHECKED"=>"1") do
        @test check_truth(other, d) == "UNCHECKED reference"
    end
end
@test_throws ErrorException check_truth(Base.structdiff(reference, (; singular_order=2)), d)

# Opposite driver orientations describe different acoustic problems even when
# the interface map and the dominant entries of the coupled rest matrix match.
motion = sparse([1], [1], [1.0im], size(d.bem_motion_flux)...)
with_motion = merge(d, (; bem_motion_flux=motion))
@test check_truth(merge(reference, (; bem_motion_flux=motion)), with_motion) == "verified Float64 dump"
@test_throws ErrorException check_truth(merge(reference, (; bem_motion_flux=-motion)), with_motion)
