from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import format_self_weight_beam_result, run_self_weight_beam_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/self_weight_beam.vtk"))
    parser.add_argument("--nx", type=int, default=8)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    args = parser.parse_args()

    result = run_self_weight_beam_demo(args.output, nx=args.nx, ny=args.ny, nz=args.nz)
    print(format_self_weight_beam_result(result))


if __name__ == "__main__":
    main()
