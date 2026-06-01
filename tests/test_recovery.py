import numpy as np

from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import element_strains, element_stresses, von_mises


def test_element_strains_recover_affine_engineering_strain():
    mesh = box_mesh(1, 1, 1)
    displacement = np.column_stack(
        (
            0.02 * mesh.nodes[:, 0] + 0.03 * mesh.nodes[:, 1],
            0.04 * mesh.nodes[:, 1] + 0.05 * mesh.nodes[:, 2],
            0.06 * mesh.nodes[:, 2] + 0.07 * mesh.nodes[:, 0],
        )
    )

    strains = element_strains(mesh, displacement)

    assert np.allclose(strains, np.array([0.02, 0.04, 0.06, 0.03, 0.05, 0.07]))


def test_element_stresses_and_von_mises_for_uniaxial_strain():
    mesh = box_mesh(1, 1, 1)
    material = IsotropicMaterial(young=10.0, poisson=0.0)
    displacement = np.column_stack((0.01 * mesh.nodes[:, 0], np.zeros((mesh.n_nodes, 2))))

    stresses = element_stresses(mesh, displacement, material)

    assert np.allclose(stresses, np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert np.allclose(von_mises(stresses), 0.1)
