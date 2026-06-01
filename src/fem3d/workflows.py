from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import TetMesh, box_mesh
from fem3d.rendering import render_convergence_png, render_deformed_surface_png
from fem3d.recovery import (
    element_strains,
    element_stresses,
    engineering_strain_tensors,
    stress_tensors,
    von_mises,
)
from fem3d.solver import (
    LinearElasticityProblem,
    SolveResult,
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


@dataclass(frozen=True)
class UniaxialTensionResult:
    output: Path
    axial_strain: float
    lateral_strain: float
    axial_stress: float
    poisson: float


@dataclass(frozen=True)
class BendingRefinementRow:
    level: int
    nx: int
    ny: int
    nz: int
    dofs: int
    tip_deflection: float
    euler_bernoulli_tip_deflection: float
    vtk: Path


@dataclass(frozen=True)
class BendingRefinementStudy:
    rows: list[BendingRefinementRow]
    csv: Path


@dataclass(frozen=True)
class SelfWeightBeamResult:
    output: Path
    max_displacement: float
    support_reaction: np.ndarray
    total_body_force: np.ndarray


@dataclass(frozen=True)
class DeformedMeshRenderResult:
    png: Path
    vtk: Path
    max_von_mises: float
    displacement_scale: float


@dataclass(frozen=True)
class ConvergencePlotResult:
    study: ConvergenceStudy
    png: Path


@dataclass(frozen=True)
class BeamSolve:
    mesh: TetMesh
    material: IsotropicMaterial
    result: SolveResult
    fixed_nodes: np.ndarray
    length: float
    width: float
    height: float


def run_beam_case(
    output: str | Path,
    nx: int = 8,
    ny: int = 2,
    nz: int = 2,
) -> BeamResult:
    output_path = Path(output)
    beam = _solve_clamped_beam(nx, ny, nz, end_traction=-1.0)
    _write_elastic_vtk(output_path, beam.mesh, beam.result.displacement, beam.material)
    return BeamResult(
        output=output_path,
        max_displacement=_max_displacement(beam.result.displacement),
        support_reaction=beam.result.reactions[beam.fixed_nodes].sum(axis=0),
    )


def run_deformed_mesh_render_demo(
    png: str | Path = Path("results/deformed_von_mises.png"),
    vtk: str | Path = Path("results/deformed_von_mises.vtk"),
    nx: int = 8,
    ny: int = 2,
    nz: int = 2,
    displacement_scale: float = 4.0,
) -> DeformedMeshRenderResult:
    png_path = Path(png)
    vtk_path = Path(vtk)
    beam = _solve_clamped_beam(nx, ny, nz, end_traction=-1.0)
    stress = element_stresses(beam.mesh, beam.result.displacement, beam.material)
    equivalent_stress = von_mises(stress)
    _write_elastic_vtk(vtk_path, beam.mesh, beam.result.displacement, beam.material)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_deformed_surface_png(
        png_path,
        beam.mesh,
        beam.result.displacement,
        equivalent_stress,
        displacement_scale,
    )
    return DeformedMeshRenderResult(
        png=png_path,
        vtk=vtk_path,
        max_von_mises=float(equivalent_stress.max()),
        displacement_scale=displacement_scale,
    )


def run_bending_refinement_demo(
    levels: list[int],
    output_dir: str | Path = Path("results/bending_refinement"),
    csv: str | Path = Path("results/bending_refinement.csv"),
) -> BendingRefinementStudy:
    output_path = Path(output_dir)
    csv_path = Path(csv)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    length = 4.0
    width = 1.0
    height = 1.0
    end_traction = -1.0
    material = IsotropicMaterial(young=1000.0, poisson=0.3)
    force = abs(end_traction) * width * height
    second_moment = width * height**3 / 12.0
    true_tip = -force * length**3 / (3.0 * material.young * second_moment)
    rows: list[BendingRefinementRow] = []
    for level in levels:
        nx = 4 * level
        ny = level
        nz = level
        beam = _solve_clamped_beam(
            nx,
            ny,
            nz,
            length=length,
            width=width,
            height=height,
            material=material,
            end_traction=end_traction,
        )
        tip_nodes = _right_face_nodes(beam.mesh, beam.length)
        tip_deflection = float(beam.result.displacement[tip_nodes, 2].mean())
        vtk = output_path / f"bending_level_{level}.vtk"
        _write_elastic_vtk(vtk, beam.mesh, beam.result.displacement, beam.material)
        rows.append(
            BendingRefinementRow(
                level=level,
                nx=nx,
                ny=ny,
                nz=nz,
                dofs=3 * beam.mesh.n_nodes,
                tip_deflection=tip_deflection,
                euler_bernoulli_tip_deflection=true_tip,
                vtk=vtk,
            )
        )
    _write_bending_refinement_csv(csv_path, rows)
    return BendingRefinementStudy(rows=rows, csv=csv_path)


def run_self_weight_beam_demo(
    output: str | Path = Path("results/self_weight_beam.vtk"),
    nx: int = 8,
    ny: int = 2,
    nz: int = 2,
) -> SelfWeightBeamResult:
    output_path = Path(output)
    body_force = np.array([0.0, 0.0, -0.4])
    beam = _solve_clamped_beam(nx, ny, nz, body_force=body_force)
    _write_elastic_vtk(output_path, beam.mesh, beam.result.displacement, beam.material)
    return SelfWeightBeamResult(
        output=output_path,
        max_displacement=_max_displacement(beam.result.displacement),
        support_reaction=beam.result.reactions[beam.fixed_nodes].sum(axis=0),
        total_body_force=body_force * beam.length * beam.width * beam.height,
    )


def run_uniaxial_tension_demo(
    output: str | Path = Path("results/uniaxial_tension.vtk"),
    nx: int = 6,
    ny: int = 3,
    nz: int = 3,
) -> UniaxialTensionResult:
    output_path = Path(output)
    length = 2.0
    axial_stress = 3.0
    material = IsotropicMaterial(young=120.0, poisson=0.25)
    axial_strain = axial_stress / material.young
    lateral_strain = -material.poisson * axial_strain
    mesh = box_mesh(nx, ny, nz, lengths=(length, 1.0, 1.0))
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))

    def exact_displacement(points: np.ndarray) -> np.ndarray:
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
        dirichlet_bcs=(DirichletBC(fixed, exact_displacement),),
        traction_loads=(TractionLoad(loaded_faces, np.array([axial_stress, 0.0, 0.0])),),
    )
    result = solve_linear_elasticity_result(problem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        output_path,
        mesh,
        result.displacement,
        cell_data=_elastic_cell_data(mesh, result.displacement, material),
    )
    return UniaxialTensionResult(
        output=output_path,
        axial_strain=axial_strain,
        lateral_strain=lateral_strain,
        axial_stress=axial_stress,
        poisson=material.poisson,
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


def run_convergence_plot_demo(
    levels: list[int],
    png: str | Path = Path("results/convergence_error.png"),
    vtk_dir: str | Path = Path("results/convergence_vtk"),
    csv: str | Path = Path("results/convergence.csv"),
) -> ConvergencePlotResult:
    png_path = Path(png)
    study = run_convergence_study(levels, vtk_dir=vtk_dir, csv=csv)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    render_convergence_png(
        png_path,
        np.array([row.h for row in study.rows], dtype=float),
        np.array([row.l2 for row in study.rows], dtype=float),
        np.array([row.h1_seminorm for row in study.rows], dtype=float),
    )
    return ConvergencePlotResult(study=study, png=png_path)


def format_beam_result(result: BeamResult) -> str:
    return "\n".join(
        (
            f"wrote {result.output}",
            f"max |u| = {result.max_displacement:.6e}",
            f"support reaction = {result.support_reaction}",
        )
    )


def format_deformed_mesh_render_result(result: DeformedMeshRenderResult) -> str:
    return "\n".join(
        (
            f"wrote {result.png}",
            f"wrote {result.vtk}",
            f"max von Mises = {result.max_von_mises:.6e}",
            f"displacement scale = {result.displacement_scale:.6e}",
        )
    )


def format_bending_refinement_study(result: BendingRefinementStudy) -> str:
    lines = ["level  mesh          dofs    tip dz       EB tip dz    ratio"]
    for row in result.rows:
        ratio = row.tip_deflection / row.euler_bernoulli_tip_deflection
        lines.append(
            f"{row.level:5d}  {row.nx}x{row.ny}x{row.nz:<5d}  {row.dofs:5d}  "
            f"{row.tip_deflection: .6e}  {row.euler_bernoulli_tip_deflection: .6e}  "
            f"{ratio: .3f}"
        )
        lines.append(f"wrote {row.vtk}")
    lines.append(f"wrote {result.csv}")
    return "\n".join(lines)


def format_self_weight_beam_result(result: SelfWeightBeamResult) -> str:
    return "\n".join(
        (
            f"wrote {result.output}",
            f"max |u| = {result.max_displacement:.6e}",
            f"total body force = {result.total_body_force}",
            f"support reaction = {result.support_reaction}",
        )
    )


def format_uniaxial_tension_result(result: UniaxialTensionResult) -> str:
    return "\n".join(
        (
            f"wrote {result.output}",
            f"axial stress = {result.axial_stress:.6e}",
            f"axial strain = {result.axial_strain:.6e}",
            f"lateral strain = {result.lateral_strain:.6e}",
            f"poisson ratio = {result.poisson:.6e}",
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


def format_convergence_plot_result(result: ConvergencePlotResult) -> str:
    lines = [format_convergence_study(result.study), f"wrote {result.png}"]
    return "\n".join(lines)


def _solve_clamped_beam(
    nx: int,
    ny: int,
    nz: int,
    length: float = 4.0,
    width: float = 1.0,
    height: float = 1.0,
    material: IsotropicMaterial | None = None,
    end_traction: float | None = None,
    body_force: np.ndarray | None = None,
) -> BeamSolve:
    mesh = box_mesh(nx, ny, nz, lengths=(length, width, height))
    fixed = _left_face_nodes(mesh)
    traction_loads = ()
    if end_traction is not None:
        loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))
        traction_loads = (TractionLoad(loaded_faces, np.array([0.0, 0.0, end_traction])),)
    beam_material = IsotropicMaterial(young=1000.0, poisson=0.3) if material is None else material
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=beam_material,
        body_force=body_force,
        dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
        traction_loads=traction_loads,
    )
    return BeamSolve(
        mesh=mesh,
        material=beam_material,
        result=solve_linear_elasticity_result(problem),
        fixed_nodes=fixed,
        length=length,
        width=width,
        height=height,
    )


def _write_elastic_vtk(
    path: Path,
    mesh: TetMesh,
    displacement: np.ndarray,
    material: IsotropicMaterial,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(path, mesh, displacement, cell_data=_elastic_cell_data(mesh, displacement, material))


def _left_face_nodes(mesh: TetMesh) -> np.ndarray:
    return mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))


def _right_face_nodes(mesh: TetMesh, length: float) -> np.ndarray:
    return mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], length))


def _max_displacement(displacement: np.ndarray) -> float:
    return float(np.linalg.norm(displacement, axis=1).max())


def _write_bending_refinement_csv(path: Path, rows: list[BendingRefinementRow]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write("level,nx,ny,nz,dofs,tip_deflection,euler_bernoulli_tip_deflection,ratio\n")
        for row in rows:
            ratio = row.tip_deflection / row.euler_bernoulli_tip_deflection
            fh.write(
                f"{row.level},{row.nx},{row.ny},{row.nz},{row.dofs},"
                f"{row.tip_deflection:.16e},{row.euler_bernoulli_tip_deflection:.16e},"
                f"{ratio:.16e}\n"
            )


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
