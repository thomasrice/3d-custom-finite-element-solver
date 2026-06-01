import numpy as np

from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import (
    element_strains,
    element_stresses,
    engineering_strain_tensors,
    stress_tensors,
    von_mises,
)


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


def test_voigt_conversions_use_correct_shear_conventions():
    strain = np.array([[1.0, 2.0, 3.0, 0.4, 0.6, 0.8]])
    stress = np.array([[1.0, 2.0, 3.0, 0.4, 0.6, 0.8]])

    strain_tensor = engineering_strain_tensors(strain)[0]
    stress_tensor = stress_tensors(stress)[0]

    assert np.allclose(strain_tensor, [[1.0, 0.2, 0.4], [0.2, 2.0, 0.3], [0.4, 0.3, 3.0]])
    assert np.allclose(stress_tensor, [[1.0, 0.4, 0.8], [0.4, 2.0, 0.6], [0.8, 0.6, 3.0]])


def test_von_mises_is_zero_for_hydrostatic_stress():
    hydrostatic = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])

    assert np.allclose(von_mises(hydrostatic), 0.0)


def test_von_mises_is_invariant_under_stress_tensor_rotation():
    stress_tensor = np.array(
        [
            [4.0, 1.2, -0.7],
            [1.2, -2.0, 0.5],
            [-0.7, 0.5, 1.0],
        ]
    )
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = rotation @ stress_tensor @ rotation.T
    stress_voigt = np.array(
        [stress_tensor[0, 0], stress_tensor[1, 1], stress_tensor[2, 2], stress_tensor[0, 1], stress_tensor[1, 2], stress_tensor[0, 2]]
    )
    rotated_voigt = np.array([rotated[0, 0], rotated[1, 1], rotated[2, 2], rotated[0, 1], rotated[1, 2], rotated[0, 2]])

    assert np.allclose(von_mises(stress_voigt), von_mises(rotated_voigt))
