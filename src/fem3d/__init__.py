"""Small 3D tetrahedral finite element solver."""

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh, box_mesh
from fem3d.solver import LinearElasticityProblem, TractionLoad, solve_linear_elasticity
from fem3d.vtk import write_vtk

__all__ = [
    "DirichletBC",
    "IsotropicMaterial",
    "LinearElasticityProblem",
    "TetMesh",
    "TractionLoad",
    "box_mesh",
    "solve_linear_elasticity",
    "write_vtk",
]
