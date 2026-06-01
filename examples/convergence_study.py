from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.solver import LinearElasticityProblem, solve_linear_elasticity
from fem3d.validation import (
    compute_error_norms,
    quadratic_body_force,
    quadratic_displacement,
    quadratic_gradient,
)
from fem3d.vtk import write_vtk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--vtk-dir", type=Path, default=None)
    args = parser.parse_args()

    material = IsotropicMaterial(young=3.0, poisson=0.25)
    rows = []
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
        norms = compute_error_norms(
            mesh,
            displacement,
            quadratic_displacement,
            quadratic_gradient,
        )
        rows.append((1.0 / n, norms.l2, norms.h1_seminorm))
        if args.vtk_dir is not None:
            args.vtk_dir.mkdir(parents=True, exist_ok=True)
            write_vtk(args.vtk_dir / f"manufactured_n{n}.vtk", mesh, displacement)

    print("h            L2 error      H1 seminorm")
    for h, l2, h1 in rows:
        print(f"{h:10.5f}  {l2:12.6e}  {h1:12.6e}")
    if len(rows) >= 2:
        hs = np.log([row[0] for row in rows])
        l2_rate = np.polyfit(hs, np.log([row[1] for row in rows]), 1)[0]
        h1_rate = np.polyfit(hs, np.log([row[2] for row in rows]), 1)[0]
        print(f"L2 rate: {l2_rate:.3f}")
        print(f"H1 rate: {h1_rate:.3f}")


if __name__ == "__main__":
    main()
