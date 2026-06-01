from __future__ import annotations

from dataclasses import dataclass, field
import warnings

import numpy as np
from scipy.sparse.linalg import MatrixRankWarning, cg, spsolve

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


def solve_linear_elasticity(
    problem: LinearElasticityProblem,
    method: str = "direct",
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    maxiter: int | None = None,
) -> np.ndarray:
    stiffness = assemble_stiffness(problem.mesh, problem.material)
    rhs = assemble_load_vector(problem)

    fixed_dofs, fixed_values = _merge_dirichlet_bcs(problem.mesh, problem.dirichlet_bcs)
    all_dofs = np.arange(3 * problem.mesh.n_nodes, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)

    solution = np.zeros(3 * problem.mesh.n_nodes, dtype=float)
    solution[fixed_dofs] = fixed_values
    _check_rigid_body_modes_constrained(problem.mesh, fixed_dofs)
    reduced_rhs = rhs[free_dofs] - stiffness[free_dofs][:, fixed_dofs] @ fixed_values
    reduced_stiffness = stiffness[free_dofs][:, free_dofs]
    if method == "direct":
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            try:
                solution[free_dofs] = spsolve(reduced_stiffness, reduced_rhs)
            except MatrixRankWarning as exc:
                raise RuntimeError(
                    "linear system is singular; check that Dirichlet constraints "
                    "remove all rigid-body modes"
                ) from exc
    elif method == "cg":
        reduced_solution, info = cg(
            reduced_stiffness,
            reduced_rhs,
            rtol=rtol,
            atol=atol,
            maxiter=maxiter,
        )
        if info != 0:
            raise RuntimeError(f"CG solver did not converge; info={info}")
        solution[free_dofs] = reduced_solution
    else:
        raise ValueError(f"unknown solver method {method!r}")
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "linear solve produced non-finite values; check mesh quality and constraints"
        )
    return solution.reshape(problem.mesh.n_nodes, 3)


def assemble_load_vector(problem: LinearElasticityProblem) -> np.ndarray:
    rhs = assemble_body_force(problem.mesh, problem.body_force)
    for load in problem.traction_loads:
        rhs += assemble_traction(problem.mesh, load.faces, load.traction)
    return rhs


def reaction_forces(problem: LinearElasticityProblem, displacement: np.ndarray) -> np.ndarray:
    """Return nodal reactions from the residual K u - f."""

    u = np.asarray(displacement, dtype=float)
    if u.shape != (problem.mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")
    stiffness = assemble_stiffness(problem.mesh, problem.material)
    residual = stiffness @ u.reshape(3 * problem.mesh.n_nodes) - assemble_load_vector(problem)
    return residual.reshape(problem.mesh.n_nodes, 3)


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


def _check_rigid_body_modes_constrained(mesh: TetMesh, fixed_dofs: np.ndarray) -> None:
    modes = np.zeros((3 * mesh.n_nodes, 6), dtype=float)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    z = mesh.nodes[:, 2]
    modes[0::3, 0] = 1.0
    modes[1::3, 1] = 1.0
    modes[2::3, 2] = 1.0
    modes[1::3, 3] = -z
    modes[2::3, 3] = y
    modes[0::3, 4] = z
    modes[2::3, 4] = -x
    modes[0::3, 5] = -y
    modes[1::3, 5] = x
    if np.linalg.matrix_rank(modes[fixed_dofs], tol=1.0e-10) < 6:
        raise RuntimeError(
            "Dirichlet constraints do not remove all rigid-body modes; "
            "add enough displacement constraints to prevent free translations and rotations"
        )
