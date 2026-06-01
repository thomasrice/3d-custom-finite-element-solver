import numpy as np
import pytest

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh, box_mesh
from fem3d.solver import LinearElasticityProblem, assemble_system


def test_conflicting_dirichlet_values_raise_clear_error():
    mesh = box_mesh(1, 1, 1)
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=IsotropicMaterial(young=10.0, poisson=0.25),
        dirichlet_bcs=(
            DirichletBC(np.array([0]), np.array([0.0, 0.0, 0.0])),
            DirichletBC(np.array([0]), np.array([1.0, 0.0, 0.0])),
        ),
    )

    with pytest.raises(ValueError, match="conflicting Dirichlet"):
        assemble_system(problem)


def test_degenerate_tetrahedron_is_rejected_by_system_assembly():
    mesh = TetMesh(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3]]),
    )
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=IsotropicMaterial(young=10.0, poisson=0.25),
        dirichlet_bcs=(DirichletBC(np.arange(4), np.zeros(3)),),
    )

    with pytest.raises(ValueError, match="inverted or degenerate"):
        assemble_system(problem)
