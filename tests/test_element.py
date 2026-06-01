import numpy as np

from fem3d.element import strain_displacement_matrix, tet_geometry


def test_unit_tet_geometry_shape_gradients():
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    volume, gradients = tet_geometry(coords)

    assert np.isclose(volume, 1.0 / 6.0)
    assert np.allclose(gradients.sum(axis=0), 0.0)
    assert np.allclose(gradients[0], [-1.0, -1.0, -1.0])
    assert strain_displacement_matrix(gradients).shape == (6, 12)
