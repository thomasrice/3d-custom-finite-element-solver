import numpy as np

from fem3d.mesh import box_mesh


def test_box_mesh_counts_and_boundary_faces():
    mesh = box_mesh(2, 1, 1, lengths=(2.0, 1.0, 1.0))

    assert mesh.nodes.shape == (12, 3)
    assert mesh.elements.shape == (12, 4)
    assert len(mesh.boundary_faces()) == 20

    right_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 2.0))
    assert len(right_faces) == 2
