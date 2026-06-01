import numpy as np

from fem3d.assembly import assemble_stiffness, assemble_traction
from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.solver import (
    LinearElasticityProblem,
    TractionLoad,
    assemble_load_vector,
    reaction_forces,
    solve_linear_elasticity,
)
from fem3d.validation import (
    compute_error_norms,
    quadratic_body_force,
    quadratic_displacement,
    quadratic_gradient,
)


def affine_displacement(points):
    return np.column_stack(
        (
            0.01 + 0.02 * points[:, 0] - 0.03 * points[:, 1],
            -0.02 + 0.04 * points[:, 1] + 0.01 * points[:, 2],
            0.03 - 0.02 * points[:, 0] + 0.05 * points[:, 2],
        )
    )


def test_constant_strain_patch_test_reproduces_affine_field():
    mesh = box_mesh(2, 2, 2)
    material = IsotropicMaterial(young=10.0, poisson=0.25)
    boundary_nodes = mesh.boundary_nodes(
        lambda x: (
            np.isclose(x[:, 0], 0.0)
            | np.isclose(x[:, 0], 1.0)
            | np.isclose(x[:, 1], 0.0)
            | np.isclose(x[:, 1], 1.0)
            | np.isclose(x[:, 2], 0.0)
            | np.isclose(x[:, 2], 1.0)
        )
    )
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(boundary_nodes, affine_displacement),),
    )

    displacement = solve_linear_elasticity(problem)

    assert np.allclose(displacement, affine_displacement(mesh.nodes), atol=1e-12)


def test_traction_assembly_distributes_constant_face_load():
    mesh = box_mesh(1, 1, 1)
    faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 1.0))
    rhs = assemble_traction(mesh, faces, np.array([2.0, 0.0, 0.0]))

    assert np.isclose(rhs[0::3].sum(), 2.0)
    assert np.allclose(rhs[1::3], 0.0)
    assert np.allclose(rhs[2::3], 0.0)


def test_traction_assembly_integrates_linear_face_load():
    mesh = box_mesh(1, 1, 1)
    faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 1.0))

    rhs = assemble_traction(
        mesh,
        faces,
        lambda x: np.column_stack((np.zeros(len(x)), np.zeros(len(x)), x[:, 1])),
    )

    assert np.isclose(rhs[2::3].sum(), 0.5)
    assert np.allclose(rhs[0::3], 0.0)
    assert np.allclose(rhs[1::3], 0.0)


def test_reactions_balance_applied_traction_load():
    mesh = box_mesh(2, 1, 1, lengths=(2.0, 1.0, 1.0))
    material = IsotropicMaterial(young=100.0, poisson=0.25)
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 2.0))
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
        traction_loads=(TractionLoad(loaded_faces, np.array([0.0, 0.0, -2.0])),),
    )

    displacement = solve_linear_elasticity(problem)
    reactions = reaction_forces(problem, displacement)
    applied_load = assemble_load_vector(problem).reshape(mesh.n_nodes, 3).sum(axis=0)
    support_reaction = reactions[fixed].sum(axis=0)

    assert np.allclose(support_reaction + applied_load, 0.0, atol=1e-10)


def test_stiffness_is_sparse_symmetric_and_nonzero():
    mesh = box_mesh(2, 2, 1)
    stiffness = assemble_stiffness(mesh, IsotropicMaterial(young=1.0, poisson=0.3))

    assert stiffness.shape == (3 * mesh.n_nodes, 3 * mesh.n_nodes)
    assert stiffness.nnz > mesh.n_nodes
    assert np.allclose((stiffness - stiffness.T).data, 0.0)


def test_quadratic_manufactured_solution_converges_at_expected_rates():
    material = IsotropicMaterial(young=3.0, poisson=0.25)
    errors = []
    hs = []
    for n in (2, 4, 8):
        mesh = box_mesh(n, n, n)
        boundary_nodes = mesh.boundary_nodes(
            lambda x: (
                np.isclose(x[:, 0], 0.0)
                | np.isclose(x[:, 0], 1.0)
                | np.isclose(x[:, 1], 0.0)
                | np.isclose(x[:, 1], 1.0)
                | np.isclose(x[:, 2], 0.0)
                | np.isclose(x[:, 2], 1.0)
            )
        )
        problem = LinearElasticityProblem(
            mesh=mesh,
            material=material,
            body_force=quadratic_body_force(material),
            dirichlet_bcs=(DirichletBC(boundary_nodes, quadratic_displacement),),
        )
        displacement = solve_linear_elasticity(problem)
        errors.append(
            compute_error_norms(
                mesh,
                displacement,
                quadratic_displacement,
                quadratic_gradient,
            )
        )
        hs.append(1.0 / n)

    l2_rate = np.polyfit(np.log(hs), np.log([err.l2 for err in errors]), 1)[0]
    h1_rate = np.polyfit(np.log(hs), np.log([err.h1_seminorm for err in errors]), 1)[0]

    assert l2_rate > 1.8
    assert h1_rate > 0.9
