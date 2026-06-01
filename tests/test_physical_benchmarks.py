import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import element_strains, element_stresses
from fem3d.solver import LinearElasticityProblem, TractionLoad, solve_linear_elasticity_result


def test_uniaxial_tension_recovers_hand_calculated_stress_and_strain():
    length = 2.0
    sigma = 3.0
    young = 120.0
    poisson = 0.25
    axial_strain = sigma / young
    lateral_strain = -poisson * axial_strain
    mesh = box_mesh(3, 2, 2, lengths=(length, 1.0, 1.0))
    material = IsotropicMaterial(young=young, poisson=poisson)
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))

    def exact_uniaxial_displacement(points):
        return np.column_stack(
            (
                axial_strain * points[:, 0],
                lateral_strain * points[:, 1],
                lateral_strain * points[:, 2],
            )
        )

    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(fixed, exact_uniaxial_displacement),),
        traction_loads=(TractionLoad(loaded_faces, np.array([sigma, 0.0, 0.0])),),
    )

    result = solve_linear_elasticity_result(problem)
    strains = element_strains(mesh, result.displacement)
    stresses = element_stresses(mesh, result.displacement, material)

    assert np.allclose(strains[:, 0], axial_strain, atol=1e-12)
    assert np.allclose(strains[:, 1], lateral_strain, atol=1e-12)
    assert np.allclose(strains[:, 2], lateral_strain, atol=1e-12)
    assert np.allclose(strains[:, 3:], 0.0, atol=1e-12)
    assert np.allclose(stresses, np.array([sigma, 0.0, 0.0, 0.0, 0.0, 0.0]), atol=1e-11)


def test_simple_shear_recovers_hand_calculated_shear_stress():
    gamma = 0.04
    young = 90.0
    poisson = 0.2
    material = IsotropicMaterial(young=young, poisson=poisson)
    mesh = box_mesh(2, 2, 2)
    fixed = mesh.boundary_nodes(
        lambda x: (
            np.isclose(x[:, 0], 0.0)
            | np.isclose(x[:, 0], 1.0)
            | np.isclose(x[:, 1], 0.0)
            | np.isclose(x[:, 1], 1.0)
            | np.isclose(x[:, 2], 0.0)
            | np.isclose(x[:, 2], 1.0)
        )
    )

    def exact_shear_displacement(points):
        return np.column_stack((gamma * points[:, 1], np.zeros((len(points), 2))))

    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(fixed, exact_shear_displacement),),
    )

    result = solve_linear_elasticity_result(problem)
    strains = element_strains(mesh, result.displacement)
    stresses = element_stresses(mesh, result.displacement, material)

    assert np.allclose(strains, np.array([0.0, 0.0, 0.0, gamma, 0.0, 0.0]), atol=1e-12)
    assert np.allclose(stresses, np.array([0.0, 0.0, 0.0, material.shear_mu * gamma, 0.0, 0.0]), atol=1e-12)
