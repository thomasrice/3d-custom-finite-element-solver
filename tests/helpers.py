import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh, box_mesh
from fem3d.solver import LinearElasticityProblem, TractionLoad


def boundary_nodes(mesh: TetMesh) -> np.ndarray:
    return mesh.boundary_nodes(
        lambda x: (
            np.isclose(x[:, 0], 0.0)
            | np.isclose(x[:, 0], 1.0)
            | np.isclose(x[:, 1], 0.0)
            | np.isclose(x[:, 1], 1.0)
            | np.isclose(x[:, 2], 0.0)
            | np.isclose(x[:, 2], 1.0)
        )
    )


def rigid_displacement(points: np.ndarray) -> np.ndarray:
    translation = np.array([0.03, -0.02, 0.01])
    omega = np.array([0.01, -0.02, 0.015])
    return translation + np.cross(np.tile(omega, (len(points), 1)), points)


def affine_displacement(points: np.ndarray) -> np.ndarray:
    return np.column_stack(
        (
            0.01 + 0.02 * points[:, 0] - 0.03 * points[:, 1] + 0.01 * points[:, 2],
            -0.02 + 0.04 * points[:, 1] + 0.01 * points[:, 2],
            0.03 - 0.02 * points[:, 0] + 0.05 * points[:, 2],
        )
    )


def clamped_beam_problem(
    traction: np.ndarray,
    nx: int = 4,
    ny: int = 2,
    nz: int = 2,
) -> tuple[LinearElasticityProblem, np.ndarray]:
    length = 4.0
    mesh = box_mesh(nx, ny, nz, lengths=(length, 1.0, 1.0))
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=IsotropicMaterial(young=1000.0, poisson=0.3),
        dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
        traction_loads=(TractionLoad(loaded_faces, np.asarray(traction, dtype=float)),),
    )
    return problem, fixed
