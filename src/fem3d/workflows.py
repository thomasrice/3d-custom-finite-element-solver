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
from fem3d.solver import (
    LinearElasticityProblem,
    TractionLoad,
    solve_linear_elasticity,
    solve_linear_elasticity_result,
)
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
    result = solve_linear_elasticity_result(problem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        output_path,
        mesh,
        result.displacement,
        cell_data=_elastic_cell_data(mesh, result.displacement, material),
    )
    return BeamResult(
        output=output_path,
        max_displacement=float(np.linalg.norm(result.displacement, axis=1).max()),
        support_reaction=result.reactions[fixed].sum(axis=0),
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
        boundary_nodes = _box_boundary_nodes(mesh)
        problem = LinearElasticityProblem(
            mesh=mesh,
            material=material,
            body_force=quadratic_body_force(material),
            dirichlet_bcs=(DirichletBC(boundary_nodes, quadratic_displacement),),
        )
        displacement = solve_linear_elasticity(problem)
        norms = compute_error_norms(
            mesh,
            displacement,
            quadratic_displacement,
            quadratic_gradient,
        )
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


def format_beam_result(result: BeamResult) -> str:
    return "\n".join(
        (
            f"wrote {result.output}",
            f"max |u| = {result.max_displacement:.6e}",
            f"support reaction = {result.support_reaction}",
        )
    )


def format_convergence_study(result: ConvergenceStudy) -> str:
    lines = ["h            L2 error      H1 seminorm"]
    for row in result.rows:
        lines.append(f"{row.h:10.5f}  {row.l2:12.6e}  {row.h1_seminorm:12.6e}")
    if result.l2_rate is not None and result.h1_rate is not None:
        lines.append(f"L2 rate: {result.l2_rate:.3f}")
        lines.append(f"H1 rate: {result.h1_rate:.3f}")
    if result.csv is not None:
        lines.append(f"wrote {result.csv}")
    return "\n".join(lines)


def _box_boundary_nodes(mesh) -> np.ndarray:
    max_corner = mesh.nodes.max(axis=0)
    return mesh.boundary_nodes(
        lambda x: (
            np.isclose(x[:, 0], 0.0)
            | np.isclose(x[:, 0], max_corner[0])
            | np.isclose(x[:, 1], 0.0)
            | np.isclose(x[:, 1], max_corner[1])
            | np.isclose(x[:, 2], 0.0)
            | np.isclose(x[:, 2], max_corner[2])
        )
    )


def _elastic_cell_data(mesh, displacement: np.ndarray, material: IsotropicMaterial):
    stress = element_stresses(mesh, displacement, material)
    return [
        CellTensor("strain", engineering_strain_tensors(element_strains(mesh, displacement))),
        CellTensor("stress", stress_tensors(stress)),
        CellScalar("von_mises", von_mises(stress)),
    ]
