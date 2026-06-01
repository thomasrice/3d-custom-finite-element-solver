from __future__ import annotations

import numpy as np

from fem3d.element import strain_displacement_matrix, tet_geometry
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh


def element_strains(mesh: TetMesh, displacement: np.ndarray) -> np.ndarray:
    """Return constant engineering strain per linear tetrahedron."""

    u = np.asarray(displacement, dtype=float)
    if u.shape != (mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")
    strains = np.zeros((mesh.n_elements, 6), dtype=float)
    for index, element in enumerate(mesh.elements):
        _, gradients = tet_geometry(mesh.nodes[element])
        b = strain_displacement_matrix(gradients)
        strains[index] = b @ u[element].reshape(12)
    return strains


def element_stresses(
    mesh: TetMesh,
    displacement: np.ndarray,
    material: IsotropicMaterial,
) -> np.ndarray:
    """Return constant Cauchy stress per linear tetrahedron in Voigt order."""

    return element_strains(mesh, displacement) @ material.elasticity_matrix().T


def von_mises(stress: np.ndarray) -> np.ndarray:
    """Return von Mises equivalent stress from Voigt stress components."""

    sigma = np.asarray(stress, dtype=float)
    if sigma.ndim == 1:
        sigma = sigma.reshape(1, 6)
    if sigma.ndim != 2 or sigma.shape[1] != 6:
        raise ValueError("stress must have shape (n, 6) or (6,)")
    sx, sy, sz, txy, tyz, txz = sigma.T
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + txz**2)
    )
