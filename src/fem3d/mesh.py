from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class TetMesh:
    nodes: np.ndarray
    elements: np.ndarray

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes, dtype=float)
        elements = np.asarray(self.elements, dtype=np.int64)
        if nodes.ndim != 2 or nodes.shape[1] != 3:
            raise ValueError("nodes must have shape (n_nodes, 3)")
        if elements.ndim != 2 or elements.shape[1] != 4:
            raise ValueError("elements must have shape (n_elements, 4)")
        if len(nodes) == 0:
            raise ValueError("mesh must contain at least one node")
        if len(elements) == 0:
            raise ValueError("mesh must contain at least one tetrahedron")
        if elements.min() < 0 or elements.max() >= len(nodes):
            raise ValueError("element connectivity references invalid node indices")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "elements", elements)

    @property
    def n_nodes(self) -> int:
        return self.nodes.shape[0]

    @property
    def n_elements(self) -> int:
        return self.elements.shape[0]

    def boundary_nodes(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        mask = predicate(self.nodes)
        if mask.shape != (self.n_nodes,):
            raise ValueError("boundary node predicate must return shape (n_nodes,)")
        return np.flatnonzero(mask)

    def boundary_faces(self) -> np.ndarray:
        faces: dict[tuple[int, int, int], tuple[int, int, int] | None] = {}
        for tet in self.elements:
            local_faces = (
                (tet[1], tet[2], tet[3]),
                (tet[0], tet[3], tet[2]),
                (tet[0], tet[1], tet[3]),
                (tet[0], tet[2], tet[1]),
            )
            for face in local_faces:
                key = tuple(sorted(int(i) for i in face))
                faces[key] = None if key in faces else tuple(int(i) for i in face)
        return np.array([face for face in faces.values() if face is not None], dtype=np.int64)

    def faces_on(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        boundary_faces = self.boundary_faces()
        centroids = self.nodes[boundary_faces].mean(axis=1)
        mask = predicate(centroids)
        if mask.shape != (len(boundary_faces),):
            raise ValueError("face predicate must return shape (n_boundary_faces,)")
        return boundary_faces[mask]

    def signed_volumes(self) -> np.ndarray:
        volumes = np.empty(self.n_elements, dtype=float)
        for index, element in enumerate(self.elements):
            matrix = np.ones((4, 4), dtype=float)
            matrix[:, 1:] = self.nodes[element]
            volumes[index] = np.linalg.det(matrix) / 6.0
        return volumes

    def scaled_jacobians(self) -> np.ndarray:
        quality = np.empty(self.n_elements, dtype=float)
        for index, element in enumerate(self.elements):
            coords = self.nodes[element]
            jacobian = np.column_stack((coords[1] - coords[0], coords[2] - coords[0], coords[3] - coords[0]))
            denominator = np.linalg.norm(jacobian[:, 0]) * np.linalg.norm(jacobian[:, 1]) * np.linalg.norm(jacobian[:, 2])
            quality[index] = 0.0 if denominator == 0.0 else np.linalg.det(jacobian) / denominator
        return quality

    def quality(self) -> "MeshQuality":
        signed_volumes = self.signed_volumes()
        scaled_jacobians = self.scaled_jacobians()
        return MeshQuality(
            min_signed_volume=float(np.min(signed_volumes)),
            max_signed_volume=float(np.max(signed_volumes)),
            min_scaled_jacobian=float(np.min(scaled_jacobians)),
            inverted_elements=np.flatnonzero(signed_volumes <= 0.0),
        )

    def require_valid_quality(self, min_scaled_jacobian: float = 1.0e-12) -> None:
        quality = self.quality()
        if len(quality.inverted_elements) > 0:
            bad = ", ".join(str(int(i)) for i in quality.inverted_elements[:10])
            raise ValueError(f"mesh contains inverted or degenerate tetrahedra: {bad}")
        if quality.min_scaled_jacobian < min_scaled_jacobian:
            raise ValueError(
                "mesh contains near-degenerate tetrahedra: "
                f"min scaled Jacobian {quality.min_scaled_jacobian:.3e}"
            )


@dataclass(frozen=True)
class MeshQuality:
    min_signed_volume: float
    max_signed_volume: float
    min_scaled_jacobian: float
    inverted_elements: np.ndarray


def box_mesh(
    nx: int,
    ny: int,
    nz: int,
    lengths: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> TetMesh:
    """Create a structured tetrahedral mesh for a rectangular box."""

    if nx < 1 or ny < 1 or nz < 1:
        raise ValueError("nx, ny, and nz must be at least 1")
    lx, ly, lz = lengths
    if lx <= 0.0 or ly <= 0.0 or lz <= 0.0:
        raise ValueError("box lengths must be positive")

    xs = np.linspace(0.0, lx, nx + 1)
    ys = np.linspace(0.0, ly, ny + 1)
    zs = np.linspace(0.0, lz, nz + 1)
    nodes = np.array([[x, y, z] for z in zs for y in ys for x in xs], dtype=float)

    def node_id(i: int, j: int, k: int) -> int:
        return k * (ny + 1) * (nx + 1) + j * (nx + 1) + i

    elements: list[list[int]] = []
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                v000 = node_id(i, j, k)
                v100 = node_id(i + 1, j, k)
                v010 = node_id(i, j + 1, k)
                v110 = node_id(i + 1, j + 1, k)
                v001 = node_id(i, j, k + 1)
                v101 = node_id(i + 1, j, k + 1)
                v011 = node_id(i, j + 1, k + 1)
                v111 = node_id(i + 1, j + 1, k + 1)
                elements.extend(
                    [
                        [v000, v100, v110, v111],
                        [v000, v110, v010, v111],
                        [v000, v010, v011, v111],
                        [v000, v011, v001, v111],
                        [v000, v001, v101, v111],
                        [v000, v101, v100, v111],
                    ]
                )
    return TetMesh(nodes=nodes, elements=np.asarray(elements, dtype=np.int64))
