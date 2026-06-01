from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import format_deformed_mesh_render_result, run_deformed_mesh_render_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--png", type=Path, default=Path("results/deformed_von_mises.png"))
    parser.add_argument("--vtk", type=Path, default=Path("results/deformed_von_mises.vtk"))
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--scale", type=float, default=4.0)
    args = parser.parse_args()

    result = run_deformed_mesh_render_demo(
        args.png,
        args.vtk,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        displacement_scale=args.scale,
    )
    print(format_deformed_mesh_render_result(result))


if __name__ == "__main__":
    main()
