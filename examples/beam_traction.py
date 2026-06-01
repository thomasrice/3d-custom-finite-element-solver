from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import run_beam_case


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("beam_traction.vtk"))
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    args = parser.parse_args()

    result = run_beam_case(args.output, nx=args.nx, ny=args.ny, nz=args.nz)
    print(f"wrote {result.output}")
    print(f"max |u| = {result.max_displacement:.6e}")
    print(f"support reaction = {result.support_reaction}")


if __name__ == "__main__":
    main()
