from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import element_strains, element_stresses, von_mises
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
from fem3d.vtk import write_vtk


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fem3d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    beam = subparsers.add_parser("beam")
    beam.add_argument("--output", type=Path, default=Path("beam_traction.vtk"))
    beam.add_argument("--nx", type=int, default=8)
    beam.add_argument("--ny", type=int, default=2)
    beam.add_argument("--nz", type=int, default=2)
    beam.set_defaults(func=_run_beam)

    convergence = subparsers.add_parser("convergence")
    convergence.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    convergence.add_argument("--vtk-dir", type=Path, default=None)
    convergence.add_argument("--csv", type=Path, default=None)
    convergence.set_defaults(func=_run_convergence)

    args = parser.parse_args(argv)
    args.func(args)


def _run_beam(args: argparse.Namespace) -> None:
    length = 4.0
    mesh = box_mesh(args.nx, args.ny, args.nz, lengths=(length, 1.0, 1.0))
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_vtk(
        args.output,
        mesh,
        displacement,
        cell_data={
            "strain": element_strains(mesh, displacement),
            "stress": stress,
            "von_mises": von_mises(stress),
        },
    )
    print(f"wrote {args.output}")
    print(f"max |u| = {np.linalg.norm(displacement, axis=1).max():.6e}")
    print(f"support reaction = {reactions[fixed].sum(axis=0)}")


def _run_convergence(args: argparse.Namespace) -> None:
    material = IsotropicMaterial(young=3.0, poisson=0.25)
    rows: list[ConvergenceRow] = []
    for n in args.levels:
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
        if args.vtk_dir is not None:
            args.vtk_dir.mkdir(parents=True, exist_ok=True)
            write_vtk(args.vtk_dir / f"manufactured_n{n}.vtk", mesh, displacement)

    print("h            L2 error      H1 seminorm")
    for row in rows:
        print(f"{row.h:10.5f}  {row.l2:12.6e}  {row.h1_seminorm:12.6e}")
    if len(rows) >= 2:
        l2_rate, h1_rate = convergence_rates(rows)
        print(f"L2 rate: {l2_rate:.3f}")
        print(f"H1 rate: {h1_rate:.3f}")
    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        write_convergence_csv(args.csv, rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()
