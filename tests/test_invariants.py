import numpy as np

from fem3d.assembly import assemble_stiffness
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.solver import assemble_system, solve_linear_elasticity_result

from helpers import clamped_beam_problem


def test_global_equilibrium_balances_applied_loads_and_support_reactions():
    problem, fixed = clamped_beam_problem(np.array([0.0, 0.0, -1.0]))

    system = assemble_system(problem)
    result = solve_linear_elasticity_result(problem)
    applied_load = system.rhs.reshape(problem.mesh.n_nodes, 3).sum(axis=0)
    support_reaction = result.reactions[fixed].sum(axis=0)

    assert np.allclose(support_reaction + applied_load, 0.0, atol=1e-10)


def test_linear_elasticity_solution_obeys_scaling_and_superposition():
    f1 = np.array([0.0, 0.0, -1.0])
    f2 = np.array([0.0, 0.5, 0.0])

    u1 = solve_linear_elasticity_result(clamped_beam_problem(f1)[0]).displacement
    u2 = solve_linear_elasticity_result(clamped_beam_problem(f2)[0]).displacement
    u_2f1 = solve_linear_elasticity_result(clamped_beam_problem(2.0 * f1)[0]).displacement
    u_f1_plus_f2 = solve_linear_elasticity_result(clamped_beam_problem(f1 + f2)[0]).displacement

    assert np.allclose(u_2f1, 2.0 * u1, rtol=1e-10, atol=1e-12)
    assert np.allclose(u_f1_plus_f2, u1 + u2, rtol=1e-10, atol=1e-12)


def test_unconstrained_stiffness_has_six_rigid_body_null_modes():
    mesh = box_mesh(1, 1, 1)
    stiffness = assemble_stiffness(mesh, IsotropicMaterial(young=10.0, poisson=0.25)).toarray()

    eigenvalues = np.linalg.eigvalsh(stiffness)
    near_zero = np.abs(eigenvalues) < 1e-9 * eigenvalues[-1]

    assert np.count_nonzero(near_zero) == 6
