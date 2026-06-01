from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from fem3d.mesh import TetMesh


@dataclass(frozen=True)
class CellScalar:
    name: str
    values: np.ndarray


@dataclass(frozen=True)
class CellVector:
    name: str
    values: np.ndarray


@dataclass(frozen=True)
class CellTensor:
    name: str
    values: np.ndarray


CellField = CellScalar | CellVector | CellTensor


def write_vtk(
    path: str | Path,
    mesh: TetMesh,
    displacement: np.ndarray | None = None,
    cell_data: Iterable[CellField] = (),
) -> None:
    """Write a legacy ASCII VTK unstructured grid readable by ParaView."""

    path = Path(path)
    displacement_array = None if displacement is None else np.asarray(displacement, dtype=float)
    if displacement_array is not None and displacement_array.shape != (mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")
    checked_cell_data = [_check_cell_field(mesh, field) for field in cell_data]

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
        if checked_cell_data:
            fh.write(f"CELL_DATA {mesh.n_elements}\n")
            for field in checked_cell_data:
                _write_cell_field(fh, field)


def _check_cell_field(mesh: TetMesh, field: CellField) -> CellField:
    if not field.name.replace("_", "").isalnum():
        raise ValueError(f"invalid VTK cell data name {field.name!r}")
    values = np.asarray(field.values, dtype=float)
    if isinstance(field, CellScalar):
        if values.shape != (mesh.n_elements,):
            raise ValueError("cell scalar values must have shape (n_elements,)")
        return CellScalar(field.name, values)
    if isinstance(field, CellVector):
        if values.shape != (mesh.n_elements, 3):
            raise ValueError("cell vector values must have shape (n_elements, 3)")
        return CellVector(field.name, values)
    if values.shape != (mesh.n_elements, 3, 3):
        raise ValueError("cell tensor values must have shape (n_elements, 3, 3)")
    return CellTensor(field.name, values)


def _write_cell_field(fh, field: CellField) -> None:
    if isinstance(field, CellScalar):
        fh.write(f"SCALARS {field.name} float 1\n")
        fh.write("LOOKUP_TABLE default\n")
        for value in field.values:
            fh.write(f"{value:.16g}\n")
    elif isinstance(field, CellVector):
        fh.write(f"VECTORS {field.name} float\n")
        for row in field.values:
            fh.write(f"{row[0]:.16g} {row[1]:.16g} {row[2]:.16g}\n")
    else:
        fh.write(f"TENSORS {field.name} float\n")
        for tensor in field.values:
            for row in tensor:
                fh.write(f"{row[0]:.16g} {row[1]:.16g} {row[2]:.16g}\n")
