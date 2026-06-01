import numpy as np

from fem3d.mesh import box_mesh
from fem3d.vtk import CellScalar, CellTensor, write_vtk


def test_write_vtk_legacy_unstructured_grid(tmp_path):
    mesh = box_mesh(1, 1, 1)
    displacement = np.zeros((mesh.n_nodes, 3))
    cell_data = [
        CellTensor("stress", np.zeros((mesh.n_elements, 3, 3))),
        CellScalar("von_mises", np.ones(mesh.n_elements)),
    ]
    path = tmp_path / "result.vtk"

    write_vtk(path, mesh, displacement, cell_data=cell_data)

    text = path.read_text(encoding="utf-8")
    assert "DATASET UNSTRUCTURED_GRID" in text
    assert f"POINTS {mesh.n_nodes} float" in text
    assert f"CELLS {mesh.n_elements} {mesh.n_elements * 5}" in text
    assert "VECTORS displacement float" in text
    assert f"CELL_DATA {mesh.n_elements}" in text
    assert "TENSORS stress float" in text
    assert "SCALARS von_mises float 1" in text


def test_write_vtk_requires_explicit_tensor_shape(tmp_path):
    mesh = box_mesh(1, 1, 1)
    path = tmp_path / "result.vtk"

    with np.testing.assert_raises(ValueError):
        write_vtk(path, mesh, cell_data=[CellTensor("strain", np.zeros((mesh.n_elements, 6)))])
