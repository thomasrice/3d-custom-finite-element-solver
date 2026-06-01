from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.boundary import DirichletBC
from fem3d.material import IsotropicMaterial
from fem3d.mesh import box_mesh
from fem3d.recovery import element_strains, element_stresses, von_mises
from fem3d.solver import LinearElasticityProblem, TractionLoad, reaction_forces, solve_linear_elasticity
from fem3d.vtk import write_vtk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("beam_traction.vtk"))
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()
