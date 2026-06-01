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
    stress = element_stresses(mesh, result.displacement, material)
    equivalent_stress = von_mises(stress)
    vtk_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        vtk_path,
        mesh,
        result.displacement,
        cell_data=_elastic_cell_data(mesh, result.displacement, material),
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    _render_deformed_surface_png(
        png_path,
        mesh,
        result.displacement,
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
        mesh = box_mesh(nx, ny, nz, lengths=(length, width, height))
        fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
        loaded_faces = mesh.faces_on(lambda x: np.isclose(x[:, 0], length))
        problem = LinearElasticityProblem(
            mesh=mesh,
            material=material,
            dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
            traction_loads=(TractionLoad(loaded_faces, np.array([0.0, 0.0, end_traction])),),
        )
        result = solve_linear_elasticity_result(problem)
        tip_nodes = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], length))
        tip_deflection = float(result.displacement[tip_nodes, 2].mean())
        vtk = output_path / f"bending_level_{level}.vtk"
        write_vtk(
            vtk,
            mesh,
            result.displacement,
            cell_data=_elastic_cell_data(mesh, result.displacement, material),
        )
        rows.append(
            BendingRefinementRow(
                level=level,
                nx=nx,
                ny=ny,
                nz=nz,
                dofs=3 * mesh.n_nodes,
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
    length = 4.0
    width = 1.0
    height = 1.0
    body_force = np.array([0.0, 0.0, -0.4])
    mesh = box_mesh(nx, ny, nz, lengths=(length, width, height))
    fixed = mesh.boundary_nodes(lambda x: np.isclose(x[:, 0], 0.0))
    material = IsotropicMaterial(young=1000.0, poisson=0.3)
    problem = LinearElasticityProblem(
        mesh=mesh,
        material=material,
        body_force=body_force,
        dirichlet_bcs=(DirichletBC(fixed, np.zeros(3)),),
    )
    result = solve_linear_elasticity_result(problem)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        output_path,
        mesh,
        result.displacement,
        cell_data=_elastic_cell_data(mesh, result.displacement, material),
    )
    return SelfWeightBeamResult(
        output=output_path,
        max_displacement=float(np.linalg.norm(result.displacement, axis=1).max()),
        support_reaction=result.reactions[fixed].sum(axis=0),
        total_body_force=body_force * length * width * height,
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


def _render_deformed_surface_png(
    path: Path,
    mesh,
    displacement: np.ndarray,
    cell_values: np.ndarray,
    displacement_scale: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    deformed = mesh.nodes + displacement_scale * displacement
    faces, values = _boundary_faces_with_cell_values(mesh, cell_values)
    polygons = [deformed[face] for face in faces]
    collection = Poly3DCollection(polygons, linewidths=0.25, edgecolors="0.25")
    collection.set_array(values)
    collection.set_cmap("viridis")
    collection.set_clim(float(values.min()), float(values.max()))

    fig = plt.figure(figsize=(8, 5), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(collection)
    _set_equal_3d_axes(ax, deformed)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    colorbar = fig.colorbar(collection, ax=ax, shrink=0.7, pad=0.02)
    colorbar.set_label("von Mises stress")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _boundary_faces_with_cell_values(mesh, cell_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    candidates: dict[tuple[int, int, int], tuple[tuple[int, int, int], float] | None] = {}
    for element_index, tet in enumerate(mesh.elements):
        local_faces = (
            (tet[1], tet[2], tet[3]),
            (tet[0], tet[3], tet[2]),
            (tet[0], tet[1], tet[3]),
            (tet[0], tet[2], tet[1]),
        )
        for face in local_faces:
            key = tuple(sorted(int(i) for i in face))
            if key in candidates:
                candidates[key] = None
            else:
                candidates[key] = (tuple(int(i) for i in face), float(cell_values[element_index]))
    boundary = [entry for entry in candidates.values() if entry is not None]
    return (
        np.array([face for face, _ in boundary], dtype=np.int64),
        np.array([value for _, value in boundary], dtype=float),
    )


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float((maxs - mins).max())
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


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
