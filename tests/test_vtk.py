import numpy as np

from fem3d.mesh import box_mesh
from fem3d.vtk import write_vtk


def test_write_vtk_legacy_unstructured_grid(tmp_path):
    mesh = box_mesh(1, 1, 1)
    displacement = np.zeros((mesh.n_nodes, 3))
    path = tmp_path / "result.vtk"

    write_vtk(path, mesh, displacement)

    text = path.read_text(encoding="utf-8")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert f"POINTS {mesh.n_nodes} float" in text
    assert f"CELLS {mesh.n_elements} {mesh.n_elements * 5}" in text
    assert "VECTORS displacement float" in text
