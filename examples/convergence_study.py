from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import run_convergence_study


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--vtk-dir", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    result = run_convergence_study(args.levels, vtk_dir=args.vtk_dir, csv=args.csv)
    print("h            L2 error      H1 seminorm")
    for row in result.rows:
        print(f"{row.h:10.5f}  {row.l2:12.6e}  {row.h1_seminorm:12.6e}")
    if result.l2_rate is not None and result.h1_rate is not None:
        print(f"L2 rate: {result.l2_rate:.3f}")
        print(f"H1 rate: {result.h1_rate:.3f}")
    if result.csv is not None:
        print(f"wrote {result.csv}")


if __name__ == "__main__":
    main()
