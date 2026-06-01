from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import spsolve

from fem3d.assembly import assemble_body_force, assemble_stiffness, assemble_traction
from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh


@dataclass(frozen=True)
class TractionLoad:
    faces: np.ndarray
    traction: object


@dataclass(frozen=True)
class LinearElasticityProblem:
    mesh: TetMesh
    material: IsotropicMaterial
    dirichlet_bcs: tuple[DirichletBC, ...]
    body_force: object | None = None
    traction_loads: tuple[TractionLoad, ...] = field(default_factory=tuple)


def solve_linear_elasticity(problem: LinearElasticityProblem) -> np.ndarray:
    stiffness = assemble_stiffness(problem.mesh, problem.material)
    rhs = assemble_body_force(problem.mesh, problem.body_force)
    for load in problem.traction_loads:
        rhs += assemble_traction(problem.mesh, load.faces, load.traction)

    fixed_dofs, fixed_values = _merge_dirichlet_bcs(problem.mesh, problem.dirichlet_bcs)
    all_dofs = np.arange(3 * problem.mesh.n_nodes, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)

    solution = np.zeros(3 * problem.mesh.n_nodes, dtype=float)
    solution[fixed_dofs] = fixed_values
    reduced_rhs = rhs[free_dofs] - stiffness[free_dofs][:, fixed_dofs] @ fixed_values
    solution[free_dofs] = spsolve(stiffness[free_dofs][:, free_dofs], reduced_rhs)
    return solution.reshape(problem.mesh.n_nodes, 3)


def _merge_dirichlet_bcs(mesh: TetMesh, bcs: tuple[DirichletBC, ...]) -> tuple[np.ndarray, np.ndarray]:
    prescribed: dict[int, float] = {}
    for bc in bcs:
        dofs, values = bc.prescribed_dofs(mesh.nodes)
        for dof, value in zip(dofs, values, strict=True):
            old = prescribed.get(int(dof))
            if old is not None and not np.isclose(old, value):
                raise ValueError(f"conflicting Dirichlet values for dof {dof}")
            prescribed[int(dof)] = float(value)
    if not prescribed:
        raise ValueError("at least one Dirichlet boundary condition is required")
    fixed_dofs = np.array(sorted(prescribed), dtype=np.int64)
    fixed_values = np.array([prescribed[int(dof)] for dof in fixed_dofs], dtype=float)
    return fixed_dofs, fixed_values
