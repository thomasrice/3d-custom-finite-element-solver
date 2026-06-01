from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, cg, spsolve

from fem3d.assembly import assemble_body_force, assemble_stiffness, assemble_traction
from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh
from fem3d.types import VectorValue


@dataclass(frozen=True)
class TractionLoad:
    faces: np.ndarray
    traction: VectorValue


@dataclass(frozen=True)
class SolverOptions:
    method: Literal["direct", "cg"] = "direct"
    rtol: float = 1.0e-10
    atol: float = 0.0
    maxiter: int | None = None

    @classmethod
    def direct(cls) -> "SolverOptions":
        return cls(method="direct")

    @classmethod
    def cg(
        cls,
        rtol: float = 1.0e-10,
        atol: float = 0.0,
        maxiter: int | None = None,
    ) -> "SolverOptions":
        return cls(method="cg", rtol=rtol, atol=atol, maxiter=maxiter)


@dataclass(frozen=True)
class LinearElasticityProblem:
    mesh: TetMesh
    material: IsotropicMaterial
    dirichlet_bcs: tuple[DirichletBC, ...]
    body_force: VectorValue | None = None
    traction_loads: tuple[TractionLoad, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LinearSystem:
    problem: LinearElasticityProblem
    stiffness: csr_matrix
    rhs: np.ndarray
    fixed_dofs: np.ndarray
    fixed_values: np.ndarray
    free_dofs: np.ndarray


@dataclass(frozen=True)
class SolveResult:
    displacement: np.ndarray
    reactions: np.ndarray
    residual: np.ndarray


def solve_linear_elasticity(
    problem: LinearElasticityProblem,
    solver: SolverOptions | None = None,
) -> np.ndarray:
    return solve_linear_elasticity_result(problem, solver).displacement


def solve_linear_elasticity_result(
    problem: LinearElasticityProblem,
    solver: SolverOptions | None = None,
) -> SolveResult:
    return solve_system(assemble_system(problem), solver)


def assemble_system(problem: LinearElasticityProblem) -> LinearSystem:
    problem.mesh.require_valid_quality()
    stiffness = assemble_stiffness(problem.mesh, problem.material)
    rhs = _assemble_load_vector(problem)
    fixed_dofs, fixed_values = _merge_dirichlet_bcs(problem.mesh, problem.dirichlet_bcs)
    all_dofs = np.arange(3 * problem.mesh.n_nodes, dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=True)
    _check_rigid_body_modes_constrained(problem.mesh, fixed_dofs)
    return LinearSystem(
        problem=problem,
        stiffness=stiffness,
        rhs=rhs,
        fixed_dofs=fixed_dofs,
        fixed_values=fixed_values,
        free_dofs=free_dofs,
    )


def solve_system(
    system: LinearSystem,
    solver: SolverOptions | None = None,
) -> SolveResult:
    options = SolverOptions.direct() if solver is None else solver
    mesh = system.problem.mesh
    solution = np.zeros(3 * mesh.n_nodes, dtype=float)
    solution[system.fixed_dofs] = system.fixed_values
    reduced_rhs = (
        system.rhs[system.free_dofs]
        - system.stiffness[system.free_dofs][:, system.fixed_dofs] @ system.fixed_values
    )
    reduced_stiffness = system.stiffness[system.free_dofs][:, system.free_dofs]
    if options.method == "direct":
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=MatrixRankWarning)
            try:
                solution[system.free_dofs] = spsolve(reduced_stiffness, reduced_rhs)
            except MatrixRankWarning as exc:
                raise RuntimeError(
                    "linear system is singular; check that Dirichlet constraints "
                    "remove all rigid-body modes"
                ) from exc
    elif options.method == "cg":
        reduced_solution, info = cg(
            reduced_stiffness,
            reduced_rhs,
            rtol=options.rtol,
            atol=options.atol,
            maxiter=options.maxiter,
        )
        if info != 0:
            raise RuntimeError(f"CG solver did not converge; info={info}")
        solution[system.free_dofs] = reduced_solution
    else:
        raise ValueError(f"unknown solver method {options.method!r}")
    if not np.all(np.isfinite(solution)):
        raise RuntimeError(
            "linear solve produced non-finite values; check mesh quality and constraints"
        )
    residual = system.stiffness @ solution - system.rhs
    return SolveResult(
        displacement=solution.reshape(mesh.n_nodes, 3),
        reactions=residual.reshape(mesh.n_nodes, 3),
        residual=residual,
    )


def _assemble_load_vector(problem: LinearElasticityProblem) -> np.ndarray:
    rhs = assemble_body_force(problem.mesh, problem.body_force)
    for load in problem.traction_loads:
        rhs += assemble_traction(problem.mesh, load.faces, load.traction)
    return rhs


def reaction_forces(
    system: LinearSystem,
    displacement: np.ndarray,
) -> np.ndarray:
    """Return nodal reactions from the residual K u - f."""

    u = np.asarray(displacement, dtype=float)
    mesh = system.problem.mesh
    if u.shape != (mesh.n_nodes, 3):
        raise ValueError("displacement must have shape (n_nodes, 3)")
    residual = system.stiffness @ u.reshape(3 * mesh.n_nodes) - system.rhs
    return residual.reshape(mesh.n_nodes, 3)


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
