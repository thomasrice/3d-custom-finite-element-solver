import numpy as np
import pytest

from fem3d.assembly import assemble_stiffness, assemble_traction
from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.solver import (
    LinearElasticityProblem,
    SolverOptions,
    TractionLoad,
    assemble_system,
    reaction_forces,
    solve_linear_elasticity,
    solve_linear_elasticity_result,
)
from fem3d.validation import (
    compute_error_norms,
    quadratic_body_force,
    quadratic_displacement,
    quadratic_gradient,
)

from helpers import boundary_nodes, clamped_beam_problem


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
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(boundary_nodes(mesh), affine_displacement),),
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
    problem, fixed = clamped_beam_problem(np.array([0.0, 0.0, -2.0]), nx=2, ny=1, nz=1)
    mesh = problem.mesh

    displacement = solve_linear_elasticity(problem)
    system = assemble_system(problem)
    reactions = reaction_forces(system, displacement)
    applied_load = system.rhs.reshape(mesh.n_nodes, 3).sum(axis=0)
    support_reaction = reactions[fixed].sum(axis=0)

    assert np.allclose(support_reaction + applied_load, 0.0, atol=1e-10)


def test_solve_result_returns_reactions_from_shared_system():
    problem, _ = clamped_beam_problem(np.array([0.0, 0.0, -2.0]), nx=2, ny=1, nz=1)

    result = solve_linear_elasticity_result(problem)

    system = assemble_system(problem)
    assert np.allclose(result.reactions, reaction_forces(system, result.displacement))
    assert np.allclose(result.reactions.reshape(-1), result.residual)


def test_cg_solver_matches_direct_solver():
    mesh = box_mesh(2, 2, 2)
    material = IsotropicMaterial(young=10.0, poisson=0.25)
    fixed_nodes = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 1.0))
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(fixed_nodes, np.zeros(3)),),
        traction_loads=(TractionLoad(loaded_faces, np.array([0.0, 0.0, -1.0])),),
    )

    direct = solve_linear_elasticity(problem)
    iterative = solve_linear_elasticity(problem, solver=SolverOptions.cg(rtol=1.0e-12))

    assert np.allclose(iterative, direct, atol=1e-10)


def test_underconstrained_problem_raises_clear_error():
    mesh = box_mesh(1, 1, 1)
    material = IsotropicMaterial(young=10.0, poisson=0.25)
    one_node = np.array([0])
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(one_node, np.zeros(3)),),
    )

    with pytest.raises(RuntimeError, match="rigid-body modes"):
        solve_linear_elasticity(problem)


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
        problem = LinearElasticityProblem(
            mesh=mesh,
            material=material,
            body_force=quadratic_body_force(material),
            dirichlet_bcs=(DirichletBC(boundary_nodes(mesh), quadratic_displacement),),
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
