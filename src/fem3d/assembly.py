from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from fem3d.element import element_stiffness, tet_geometry
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh
from fem3d.types import VectorValue


TET_QUAD_BARY = np.array(
    [
        [0.5854101966249685, 0.1381966011250105, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.5854101966249685, 0.1381966011250105, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.5854101966249685, 0.1381966011250105],
        [0.1381966011250105, 0.1381966011250105, 0.1381966011250105, 0.5854101966249685],
    ],
    dtype=float,
)
TET_QUAD_WEIGHTS = np.full(4, 0.25, dtype=float)
TRI_QUAD_BARY = np.array(
    [
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
        [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
        [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
    ],
    dtype=float,
)
TRI_QUAD_WEIGHTS = np.full(3, 1.0 / 3.0, dtype=float)


def dofs_for_element(element: np.ndarray) -> np.ndarray:
    dofs = np.empty(12, dtype=np.int64)
    for a, node in enumerate(element):
        dofs[3 * a : 3 * a + 3] = (3 * node, 3 * node + 1, 3 * node + 2)
    return dofs


def assemble_stiffness(mesh: TetMesh, material: IsotropicMaterial) -> csr_matrix:
    mesh.require_valid_quality()
    elasticity = material.elasticity_matrix()
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for element in mesh.elements:
        coords = mesh.nodes[element]
        ke = element_stiffness(coords, elasticity)
        dofs = dofs_for_element(element)
        rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(rr.ravel())
        cols.extend(cc.ravel())
        data.extend(ke.ravel())
    n_dof = 3 * mesh.n_nodes
    return coo_matrix((data, (rows, cols)), shape=(n_dof, n_dof)).tocsr()


def assemble_body_force(mesh: TetMesh, body_force: VectorValue | None) -> np.ndarray:
    mesh.require_valid_quality()
    rhs = np.zeros(3 * mesh.n_nodes, dtype=float)
    if body_force is None:
        return rhs
    for element in mesh.elements:
        coords = mesh.nodes[element]
        volume, _ = tet_geometry(coords)
        fe = np.zeros((4, 3), dtype=float)
        for bary, weight in zip(TET_QUAD_BARY, TET_QUAD_WEIGHTS, strict=True):
            point = bary @ coords
            value = _vector_value(body_force, point.reshape(1, 3))[0]
            fe += weight * volume * bary[:, None] * value[None, :]
        for local, node in enumerate(element):
            rhs[3 * node : 3 * node + 3] += fe[local]
    return rhs


def assemble_traction(
    mesh: TetMesh,
    faces: np.ndarray,
    traction: VectorValue,
) -> np.ndarray:
    rhs = np.zeros(3 * mesh.n_nodes, dtype=float)
    face_array = np.asarray(faces, dtype=np.int64)
    for face in face_array:
        coords = mesh.nodes[face]
        area = 0.5 * np.linalg.norm(np.cross(coords[1] - coords[0], coords[2] - coords[0]))
        for bary, weight in zip(TRI_QUAD_BARY, TRI_QUAD_WEIGHTS, strict=True):
            point = bary @ coords
            value = _vector_value(traction, point.reshape(1, 3))[0]
            for local, node in enumerate(face):
                rhs[3 * node : 3 * node + 3] += weight * area * bary[local] * value
    return rhs


def _vector_value(fn_or_vector: VectorValue, points: np.ndarray) -> np.ndarray:
    if callable(fn_or_vector):
        value = np.asarray(fn_or_vector(points), dtype=float)
    else:
        value = np.asarray(fn_or_vector, dtype=float)
    if value.shape == (3,):
        value = np.tile(value, (len(points), 1))
    if value.shape != (len(points), 3):
        raise ValueError("vector value must have shape (n_points, 3) or (3,)")
    return value
