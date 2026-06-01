from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import format_convergence_study, run_convergence_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--vtk-dir", type=Path, default=Path("results/convergence_vtk"))
    parser.add_argument("--csv", type=Path, default=Path("results/convergence.csv"))
    args = parser.parse_args()

    result = run_convergence_study(args.levels, vtk_dir=args.vtk_dir, csv=args.csv)
    print(format_convergence_study(result))


if __name__ == "__main__":
    main()
