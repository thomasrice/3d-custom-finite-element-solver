import numpy as np

from fem3d.solver import assemble_system, solve_linear_elasticity_result

from helpers import clamped_beam_problem


def test_global_equilibrium_balances_applied_loads_and_support_reactions():
    problem, fixed = clamped_beam_problem(np.array([0.0, 0.0, -1.0]))

    system = assemble_system(problem)
    result = solve_linear_elasticity_result(problem)
    applied_load = system.rhs.reshape(problem.mesh.n_nodes, 3).sum(axis=0)
    support_reaction = result.reactions[fixed].sum(axis=0)

    assert np.allclose(support_reaction + applied_load, 0.0, atol=1e-10)
