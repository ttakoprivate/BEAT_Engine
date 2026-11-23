# The production Float64 LU is the reference. Replace only its BEM operators
# with the H blocks in a second system and use the ordinary physical recovery.
function materialize_h(h)
    a = zeros(ComplexF64, h.n, h.m)
    for b in h.dense
        a[b.rows, b.cols] = b.M
    end
    for b in h.lr
        a[b.rows, b.cols] = b.U * b.V'
    end
    return a
end

function coupled_at(fem, bem, mapping, driver, radiator, frequency, tol, near)
    sys = build_condensed_coupled_system(fem, bem, mapping, frequency, 343., 1.21;
                quadrature_order=1, singular_order=1, transducers=[driver], bem_backend=:cpu)
    try
        p1, dp0 = build_p1_space(bem), build_dp0_space(bem)
        rule = triangle_rule(Float64, 1)
        cache = sys.cache.base
        op = fmm_burton_miller_operator(bem, p1, dp0, 2pi*frequency/343, rule;
                identity_p1_p1=cache.identity_p1_p1, identity_p1_dp0=cache.identity_p1_dp0,
                singular_order=1, symmetry_mode=sys.symmetry_mode, near_factor=near, eps=1e-10)
        flux = sparse(ComplexF64.(cache.interface_operators.bem_flux))
        motion = sparse(neumann_scale(1.21, 2pi*frequency) .*
                        assemble_transducer_operators(fem, bem, [driver]).bem_normal_velocity)
        active = union(rowvals(flux), rowvals(motion))
        faces = findall(el -> el.dp0_dof in active, op.elements)
        dofs = [op.elements[i].dp0_dof for i in faces]
        a, negc = build_h_operators(bem, op, p1, dp0, rule, sys.symmetry_mode, faces, tol, 48)
        ref_lu = sys.factorization
        @test ref_lu isa LU{ComplexF64}
        approximate = ref_lu.P' * (ref_lu.L * ref_lu.U)
        b = sys.bem_range
        approximate[b, b] = materialize_h(a)
        c = -materialize_h(negc)
        approximate[b, sys.flux_range] = c * flux[dofs, :]
        approximate[b, sys.mechanical_range] = c * motion[dofs, :]
        hsys = merge(sys, (; factorization=lu!(approximate)))
        drives = [(kind=:voltage, transducer_index=1, radiator_tag=0, amplitude=1.0+0im),
                  (kind=:normal_velocity, transducer_index=0, radiator_tag=radiator, amplitude=0.001+0.002im)]
        reference = solve_condensed_coupled_excitations(sys, drives)
        result = solve_condensed_coupled_excitations(hsys, drives)
        single = only(solve_condensed_coupled_excitations(hsys, drives[1:1]))
        # Choose a fixed linear combination from the reference so two complex drives
        # cancel 97% of the voltage response at a well-resolved boundary node.
        node = argmax(abs.(reference[2].bem_pressure))
        alpha = -0.97 * reference[1].bem_pressure[node] / reference[2].bem_pressure[node]
        @test abs(reference[1].bem_pressure[node] + alpha*reference[2].bem_pressure[node]) ≈
              0.03 * abs(reference[1].bem_pressure[node]) rtol=1e-12
        # Fixture regression gate, not the acceptance threshold for real models.
        limit = 1e-4
        for field in (:fem_pressure, :bem_pressure, :interface_flux, :bem_neumann, :diaphragm_velocity, :voice_coil_current)
            r1, r2 = (getproperty(x, field) for x in reference)
            h1, h2 = (getproperty(x, field) for x in result)
            for (label, h, r) in (("voltage", h1, r1), ("velocity", h2, r2),
                                  ("interference", h1 + alpha*h2, r1 + alpha*r2))
                e = errors(h, r)
                @printf("PHYSICAL %.8f %s %s max_rel=%.8g zero_count=%d max_abs_at_zero=%.8g\n",
                        frequency, label, field, e.relative, e.zeros, e.absolute)
                @test e.relative <= limit
                @test iszero(e.absolute)
            end
            @test errors(getproperty(single, field), h1).relative <= 1e-9
        end
    finally
        release_condensed_coupled_system!(sys)
    end
end

function test_coupled()
    fem = load_gmsh41_volume(joinpath(CONDENSED_FIXTURE_ROOT, "femvolume.msh"), 0.001)
    bem = load_gmsh22_with_tags(joinpath(CONDENSED_FIXTURE_ROOT, "exterior_conforming.msh"), 0.001)
    mapping = build_conforming_interface_map(fem, bem, physical_tag(fem, 2, "Interface"), 2)
    radiator = physical_tag(fem, 2, "Radiator")
    driver = ElectrodynamicTransducer{Float64}("hmatrix:reference", [radiator], [1.], [1], [-1.],
                SVector(0.,0.,1.), 2., 1, 6., 0.0005, 7., 0.015, 0.0005, 1.)
    stiffness, mass = assemble_p1_fem_matrices(fem)
    retained = union(mapping.fem_vertex_indices, findnz(assemble_transducer_operators(fem, bem, [driver]).fem_surface)[1])
    interior = setdiff(eachindex(fem.vertices), retained)
    eig = eigvals(Symmetric(Matrix(stiffness[interior, interior])), Symmetric(Matrix(mass[interior, interior])))
    positive = filter(>(1e-8 * maximum(abs, eig)), eig)
    pole = 343 * sqrt(minimum(positive)) / (2pi)
    with_baseline_cpu_coupled() do
        coupled_at(fem, bem, mapping, driver, radiator, 2000., 1e-6, 0.25)
        coupled_at(fem, bem, mapping, driver, radiator, 0.995 * pole, 1e-6, 4.)
    end
end
