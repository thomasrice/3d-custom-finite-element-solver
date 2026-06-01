from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fem3d.workflows import format_bending_refinement_study, run_bending_refinement_demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output-dir", type=Path, default=Path("results/bending_refinement"))
    parser.add_argument("--csv", type=Path, default=Path("results/bending_refinement.csv"))
    args = parser.parse_args()

    result = run_bending_refinement_demo(args.levels, output_dir=args.output_dir, csv=args.csv)
    print(format_bending_refinement_study(result))


if __name__ == "__main__":
    main()
