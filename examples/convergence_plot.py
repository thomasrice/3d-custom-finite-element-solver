from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import format_convergence_plot_result, run_convergence_plot_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--png", type=Path, default=Path("results/convergence_error.png"))
    parser.add_argument("--vtk-dir", type=Path, default=Path("results/convergence_vtk"))
    parser.add_argument("--csv", type=Path, default=Path("results/convergence.csv"))
    args = parser.parse_args()

    result = run_convergence_plot_demo(args.levels, png=args.png, vtk_dir=args.vtk_dir, csv=args.csv)
    print(format_convergence_plot_result(result))


if __name__ == "__main__":
    main()
