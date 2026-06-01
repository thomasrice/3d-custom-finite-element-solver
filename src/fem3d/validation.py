from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fem3d.element import tet_geometry
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh


@dataclass(frozen=True)
class ErrorNorms:
    l2: float
    h1_seminorm: float


@dataclass(frozen=True)
class ConvergenceRow:
    h: float
    dofs: int
    l2: float
    h1_seminorm: float


def quadratic_displacement(points: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.column_stack((x * x, y * y, z * z))


def quadratic_gradient(points: np.ndarray) -> np.ndarray:
    gradients = np.zeros((len(points), 3, 3), dtype=float)
    gradients[:, 0, 0] = 2.0 * points[:, 0]
    gradients[:, 1, 1] = 2.0 * points[:, 1]
    gradients[:, 2, 2] = 2.0 * points[:, 2]
    return gradients


def quadratic_body_force(material: IsotropicMaterial):
    lam = material.lame_lambda
    mu = material.shear_mu
    value = np.array(
        [
            -(2.0 * lam + 4.0 * mu),
            -(2.0 * lam + 4.0 * mu),
            -(2.0 * lam + 4.0 * mu),
        ],
        dtype=float,
    )

    def body(points: np.ndarray) -> np.ndarray:
        return np.tile(value, (len(points), 1))

    return body


def compute_error_norms(
    mesh: TetMesh,
    displacement: np.ndarray,
    exact_displacement,
    exact_gradient,
) -> ErrorNorms:
    u = np.asarray(displacement, dtype=float)
    if u.shape != (mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")

    # Four positive points are enough for stable rate checks on this smooth polynomial case.
    bary_points = np.array(
        [
            [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
            [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
            [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
            [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
        ],
        dtype=float,
    )
    weights = np.full(4, 0.25, dtype=float)
    l2_sq = 0.0
    h1_sq = 0.0
    for element in mesh.elements:
        coords = mesh.nodes[element]
        values = u[element]
        volume, gradients = tet_geometry(coords)
        element_gradient = values.T @ gradients
        for bary, weight in zip(bary_points, weights, strict=True):
            point = bary @ coords
            uh = bary @ values
            ue = exact_displacement(point.reshape(1, 3))[0]
            ge = exact_gradient(point.reshape(1, 3))[0]
            l2_sq += weight * volume * float(np.dot(uh - ue, uh - ue))
            grad_error = element_gradient - ge
            h1_sq += weight * volume * float(np.sum(grad_error * grad_error))
    return ErrorNorms(l2=float(np.sqrt(l2_sq)), h1_seminorm=float(np.sqrt(h1_sq)))


def convergence_rates(rows: list[ConvergenceRow]) -> tuple[float, float]:
    if len(rows) < 2:
        raise ValueError("at least two convergence rows are required")
    hs = np.log([row.h for row in rows])
    l2_rate = np.polyfit(hs, np.log([row.l2 for row in rows]), 1)[0]
    h1_rate = np.polyfit(hs, np.log([row.h1_seminorm for row in rows]), 1)[0]
    return float(l2_rate), float(h1_rate)


def write_convergence_csv(path: str | Path, rows: list[ConvergenceRow]) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        fh.write("h,dofs,l2,h1_seminorm\n")
        for row in rows:
            fh.write(f"{row.h:.16g},{row.dofs},{row.l2:.16e},{row.h1_seminorm:.16e}\n")
