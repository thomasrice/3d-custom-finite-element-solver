"""Small 3D tetrahedral finite element solver."""

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import MeshQuality, TetMesh, box_mesh
from fem3d.recovery import (
    element_strains,
    element_stresses,
    engineering_strain_tensors,
    stress_tensors,
    von_mises,
)
from fem3d.solver import (
    LinearElasticityProblem,
    LinearSystem,
    SolverOptions,
    SolveResult,
    TractionLoad,
    assemble_load_vector,
    assemble_system,
    reaction_forces,
    solve_linear_elasticity,
    solve_linear_elasticity_result,
    solve_system,
)
from fem3d.vtk import CellScalar, CellTensor, CellVector, write_vtk

__all__ = [
    "CellScalar",
    "CellTensor",
    "CellVector",
    "DirichletBC",
    "IsotropicMaterial",
    "LinearElasticityProblem",
    "LinearSystem",
    "MeshQuality",
    "SolverOptions",
    "SolveResult",
    "TetMesh",
    "TractionLoad",
    "assemble_load_vector",
    "assemble_system",
    "box_mesh",
    "element_strains",
    "element_stresses",
    "engineering_strain_tensors",
    "reaction_forces",
    "solve_linear_elasticity",
    "solve_linear_elasticity_result",
    "solve_system",
    "stress_tensors",
    "von_mises",
    "write_vtk",
]
