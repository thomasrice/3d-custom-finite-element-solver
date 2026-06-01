import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh, box_mesh
from fem3d.recovery import element_stresses
from fem3d.solver import LinearElasticityProblem, solve_linear_elasticity_result

from helpers import affine_displacement, boundary_nodes


def test_affine_patch_test_on_distorted_mesh_reproduces_exact_solution_and_stress():
    base_mesh = box_mesh(3, 3, 3)
    nodes = base_mesh.nodes.copy()
    boundary = boundary_nodes(base_mesh)
    interior_mask = np.ones(base_mesh.n_nodes, dtype=bool)
    interior_mask[boundary] = False
    rng = np.random.default_rng(20240602)
    nodes[interior_mask] += rng.uniform(-0.04, 0.04, size=(np.count_nonzero(interior_mask), 3))
    mesh = TetMesh(nodes, base_mesh.elements)
    mesh.require_valid_quality()
    material = IsotropicMaterial(young=10.0, poisson=0.25)
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(boundary_nodes(mesh), affine_displacement),),
    )

    result = solve_linear_elasticity_result(problem)
    stresses = element_stresses(mesh, result.displacement, material)

    assert np.allclose(result.displacement, affine_displacement(mesh.nodes), atol=1e-12)
    assert np.allclose(stresses, stresses[0], atol=1e-11)
