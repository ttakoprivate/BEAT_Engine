# Exercise the unmodified command-line checker in a new Julia process. The reference
# uses its public binary dump format; the files are created only in a temporary directory.
function write_dump(dir, data)
    mkpath(dir)
    open(joinpath(dir, "manifest.txt"), "w") do manifest
        function array(name, a)
            write(joinpath(dir, name * ".bin"), Array(a))
            println(manifest, name, " array ", eltype(a), " ", join(size(a), " "))
        end
        for (key, value) in pairs(data)
            name = string(key)
            if value isa SparseMatrixCSC
                i, j, v = findnz(value)
                array(name * "__I", Int64.(i))
                array(name * "__J", Int64.(j))
                array(name * "__V", ComplexF64.(v))
                println(manifest, name, " sparse ", join(size(value), " "))
            elseif value isa AbstractRange
                println(manifest, name, " range ", first(value), " ", last(value))
            elseif value isa BoundaryMesh
                array(name * "__vertices", reduce(hcat, value.vertices))
                array(name * "__faces", reduce(hcat, collect.(value.faces)))
                array(name * "__tags", value.physical_tags)
                println(manifest, name, " mesh")
            elseif value isa AbstractArray
                array(name, value)
            elseif value isa Real
                println(manifest, name, " scalar ", typeof(value), " ", value)
            else
                println(manifest, name, " text ", value)
            end
        end
    end
end

function test_dump_contract()
    mesh = _fmm_icosphere(0.1, 0)
    p1, dp0 = build_p1_space(mesh), build_dp0_space(mesh)
    np, nf = p1.global_dof_count, dp0.global_dof_count
    rule, k = triangle_rule(Float64, 2), 2pi * 2000 / 343
    pp = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :p1)
    pd = assemble_l2_identity_matrix(mesh, p1, dp0, rule, :p1, :dp0)
    flux = sparse([1], [1], [1.0+0im], nf, 1)
    dense = assemble_burton_miller_neumann_system_cpu(mesh, p1, dp0, Matrix(flux), k, rule;
            identity_p1_p1=pp, identity_p1_dp0=pd, singular_order=2)
    n = np + 3
    delta = 3 * 2.0^-24
    rest = zeros(ComplexF64, n, n)
    rest[np+1,np+1] = 1
    rest[n-1:n,n-1:n] = [1 1; 1 1+delta]
    a = copy(rest)
    a[1:np,1:np] = dense.matrix
    a[1:np,np+1:np+1] = -dense.rhs
    b = zeros(ComplexF64, n, 2)
    b[n,1] = delta * (1+im)
    b[np+1,2] = 1e-10im
    reference = a \ b
    narrowed = ComplexF64.(ComplexF32.(a) \ ComplexF32.(b))
    @test errors(narrowed[n-1:n,1], reference[n-1:n,1]).relative ≈ 0.25
    data = (; frequency_hz=2000., wavenumber=k, symmetry_mode=:off,
        regular_quadrature_order=2, singular_order=2, bem_mesh=mesh, bem_range=1:np,
        flux_range=np+1:np+1, mechanical_range=n-1:n-1, electrical_range=n:n, gamma_range=1:0,
        bem_flux=flux, bem_motion_flux=spzeros(ComplexF64,nf,1), identity_p1_p1=sparse(pp), identity_p1_dp0=sparse(pd),
        bem_lhs_rows=dense.matrix, bem_iface_rows=-dense.rhs, rhs=b, bem_backend=:cpu)
    mktempdir() do dir
        truth, mixed = joinpath(dir, "truth"), joinpath(dir, "mixed")
        write_dump(truth, (; data..., rest=sparse(rest), dense_solution=reference, model_scalar_type=:Float64,
                  assembly_eltype=:ComplexF64, solver_scalar_type=:Float64, solution_eltype=:ComplexF64, dense_solver=:lu_float64))
        write_dump(mixed, (; data..., rest=sparse(ComplexF64.(ComplexF32.(rest))), dense_solution=narrowed,
                  model_scalar_type=:Float32, assembly_eltype=:ComplexF32, solver_scalar_type=:Float32,
                  solution_eltype=:ComplexF32, dense_solver=:lu_float32))
        checker = joinpath(HM, "tests", "hmat_dump_check.jl")
        assertions = joinpath(@__DIR__, "dump_assertions.jl")
        cmd = Cmd(`$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$LOCAL -L $checker $assertions $mixed cpu 1e-6`; dir=LOCAL)
        cmd = addenv(cmd, "HMAT_TRUTH_DIR"=>truth, "HMAT_TRUTH_UNCHECKED"=>"0",
                     "HMAT_TRUTH_DISCRETISATION_STUDY"=>"0", "HMAT_EST_NEXC"=>"1",
                     "HMAT_ROWSAMPLE"=>"0", "HMAT_SOLVE"=>"0")
        out = PipeBuffer()
        ok = success(pipeline(cmd; stdout=out, stderr=out))
        report = read(out, String)
        ok || println(report)
        @test ok
        @test occursin("verified Float64 dump", report)
        @test occursin("rest and rhs taken from the checked dump", report)
        @test occursin(r"mech true 2.5e-01 est 0.0e\+00", report)
    end
end
