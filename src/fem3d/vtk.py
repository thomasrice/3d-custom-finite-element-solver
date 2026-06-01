from __future__ import annotations

from pathlib import Path

import numpy as np

from fem3d.mesh import TetMesh


def write_vtk(path: str | Path, mesh: TetMesh, displacement: np.ndarray | None = None) -> None:
    """Write a legacy ASCII VTK unstructured grid readable by ParaView."""

    path = Path(path)
    displacement_array = None if displacement is None else np.asarray(displacement, dtype=float)
    if displacement_array is not None and displacement_array.shape != (mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")

    with path.open("w", encoding="utf-8") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("custom fem3d result\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")
        fh.write(f"POINTS {mesh.n_nodes} float\n")
        for x, y, z in mesh.nodes:
            fh.write(f"{x:.16g} {y:.16g} {z:.16g}\n")
        total_cell_size = mesh.n_elements * 5
        fh.write(f"CELLS {mesh.n_elements} {total_cell_size}\n")
        for tet in mesh.elements:
            fh.write(f"4 {tet[0]} {tet[1]} {tet[2]} {tet[3]}\n")
        fh.write(f"CELL_TYPES {mesh.n_elements}\n")
        for _ in mesh.elements:
            fh.write("10\n")
        if displacement_array is not None:
            fh.write(f"POINT_DATA {mesh.n_nodes}\n")
            fh.write("VECTORS displacement float\n")
            for ux, uy, uz in displacement_array:
                fh.write(f"{ux:.16g} {uy:.16g} {uz:.16g}\n")
