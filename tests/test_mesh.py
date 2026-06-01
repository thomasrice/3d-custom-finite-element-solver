import numpy as np

import pytest

from fem3d.mesh import TetMesh, box_mesh
from fem3d.assembly import assemble_stiffness
from fem3d.material import IsotropicMaterial


def test_box_mesh_counts_and_boundary_faces():
    mesh = box_mesh(2, 1, 1, lengths=(2.0, 1.0, 1.0))

    assert mesh.nodes.shape == (12, 3)
    assert mesh.elements.shape == (12, 4)
    assert len(mesh.boundary_faces()) == 20

    right_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], 2.0))
    assert len(right_faces) == 2


def test_box_mesh_quality_is_valid():
    mesh = box_mesh(2, 2, 2)
    quality = mesh.quality()

    assert quality.min_signed_volume > 0.0
    assert quality.max_signed_volume == quality.min_signed_volume
    assert quality.min_scaled_jacobian > 0.0
    mesh.require_valid_quality()


def test_mesh_quality_detects_inverted_tet():
    mesh = TetMesh(
        nodes=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        elements=np.array([[0, 1, 2, 3]]),
    )

    quality = mesh.quality()

    assert quality.min_signed_volume < 0.0
    assert quality.inverted_elements.tolist() == [0]
    with pytest.raises(ValueError, match="inverted"):
        mesh.require_valid_quality()
    with pytest.raises(ValueError, match="inverted"):
        assemble_stiffness(mesh, IsotropicMaterial(young=1.0, poisson=0.25))
