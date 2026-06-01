from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import (
    element_strains,
    element_stresses,
    engineering_strain_tensors,
    stress_tensors,
    von_mises,
)
from fem3d.solver import LinearElasticityProblem, TractionLoad, reaction_forces, solve_linear_elasticity
from fem3d.validation import (
    ConvergenceRow,
    compute_error_norms,
    convergence_rates,
    quadratic_body_force,
    quadratic_displacement,
    quadratic_gradient,
    write_convergence_csv,
)
from fem3d.vtk import CellScalar, CellTensor, write_vtk


@dataclass(frozen=True)
class BeamResult:
    output: Path
    max_displacement: float
    support_reaction: np.ndarray


@dataclass(frozen=True)
class ConvergenceStudy:
    rows: list[ConvergenceRow]
    l2_rate: float | None
    h1_rate: float | None
    csv: Path | None = None


def run_beam_case(
    output: str | Path,
    nx: int = 8,
    ny: int = 2,
    nz: int = 2,
) -> BeamResult:
    output_path = Path(output)
    length = 4.0
    mesh = box_mesh(nx, ny, nz, lengths=(length, 1.0, 1.0))
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))
    material = IsotropicMaterial(young=1000.0, poisson=0.3)
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
        traction_loads=(TractionLoad(loaded_faces, np.array([0.0, 0.0, -1.0])),),
    )
    displacement = solve_linear_elasticity(problem)
    reactions = reaction_forces(problem, displacement)
    stress = element_stresses(mesh, displacement, material)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        output_path,
        mesh,
        displacement,
        cell_data=[
            CellTensor("strain", engineering_strain_tensors(element_strains(mesh, displacement))),
            CellTensor("stress", stress_tensors(stress)),
            CellScalar("von_mises", von_mises(stress)),
        ],
    )
    return BeamResult(
        output=output_path,
        max_displacement=float(np.linalg.norm(displacement, axis=1).max()),
        support_reaction=reactions[fixed].sum(axis=0),
    )


def run_convergence_study(
    levels: list[int],
    vtk_dir: str | Path | None = None,
    csv: str | Path | None = None,
) -> ConvergenceStudy:
    material = IsotropicMaterial(young=3.0, poisson=0.25)
    vtk_path = None if vtk_dir is None else Path(vtk_dir)
    csv_path = None if csv is None else Path(csv)
    rows: list[ConvergenceRow] = []
    for n in levels:
        mesh = box_mesh(n, n, n)
        boundary_nodes = mesh.boundary_nodes(
            lambda x: (
                np.isclose(x[:, 0], 0.0)
                | np.isclose(x[:, 0], 1.0)
                | np.isclose(x[:, 1], 0.0)
                | np.isclose(x[:, 1], 1.0)
                | np.isclose(x[:, 2], 0.0)
                | np.isclose(x[:, 2], 1.0)
            )
        )
        problem = LinearElasticityProblem(
            mesh=mesh,
            material=material,
            body_force=quadratic_body_force(material),
            dirichlet_bcs=(DirichletBC(boundary_nodes, quadratic_displacement),),
        )
        displacement = solve_linear_elasticity(problem)
        norms = compute_error_norms(mesh, displacement, quadratic_displacement, quadratic_gradient)
        rows.append(ConvergenceRow(1.0 / n, 3 * mesh.n_nodes, norms.l2, norms.h1_seminorm))
        if vtk_path is not None:
            vtk_path.mkdir(parents=True, exist_ok=True)
            write_vtk(vtk_path / f"manufactured_n{n}.vtk", mesh, displacement)

    l2_rate = None
    h1_rate = None
    if len(rows) >= 2:
        l2_rate, h1_rate = convergence_rates(rows)
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_convergence_csv(csv_path, rows)
    return ConvergenceStudy(rows=rows, l2_rate=l2_rate, h1_rate=h1_rate, csv=csv_path)
