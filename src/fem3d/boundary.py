from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fem3d.types import VectorValue


@dataclass(frozen=True)
class DirichletBC:
    node_ids: np.ndarray
    values: VectorValue
    components: tuple[int, ...] = (0, 1, 2)

    def prescribed_dofs(self, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        node_ids = np.asarray(self.node_ids, dtype=np.int64)
        if callable(self.values):
            values = np.asarray(self.values(nodes[node_ids]), dtype=float)
        else:
            values = np.asarray(self.values, dtype=float)
            if values.shape == (3,):
                values = np.tile(values, (len(node_ids), 1))
        if values.shape != (len(node_ids), 3):
            raise ValueError("Dirichlet values must have shape (n_nodes, 3) or (3,)")
        dofs: list[int] = []
        prescribed: list[float] = []
        for row, node in enumerate(node_ids):
            for component in self.components:
                dofs.append(3 * int(node) + component)
                prescribed.append(float(values[row, component]))
        return np.asarray(dofs, dtype=np.int64), np.asarray(prescribed, dtype=float)
