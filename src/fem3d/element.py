from __future__ import annotations

import numpy as np


def tet_geometry(coords: np.ndarray) -> tuple[float, np.ndarray]:
    """Return tetrahedron volume and shape-function gradients."""

    matrix = np.ones((4, 4), dtype=float)
    matrix[:, 1:] = coords
    det = np.linalg.det(matrix)
    volume = abs(det) / 6.0
    if volume <= 0.0:
        raise ValueError("degenerate tetrahedron")
    inv = np.linalg.inv(matrix)
    gradients = inv[1:, :].T
    return volume, gradients


def strain_displacement_matrix(gradients: np.ndarray) -> np.ndarray:
    b = np.zeros((6, 12), dtype=float)
    for a, (gx, gy, gz) in enumerate(gradients):
        col = 3 * a
        b[0, col] = gx
        b[1, col + 1] = gy
        b[2, col + 2] = gz
        b[3, col] = gy
        b[3, col + 1] = gx
        b[4, col + 1] = gz
        b[4, col + 2] = gy
        b[5, col] = gz
        b[5, col + 2] = gx
    return b


def element_stiffness(coords: np.ndarray, elasticity: np.ndarray) -> np.ndarray:
    volume, gradients = tet_geometry(coords)
    b = strain_displacement_matrix(gradients)
    return volume * (b.T @ elasticity @ b)
