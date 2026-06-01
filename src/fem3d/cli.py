from __future__ import annotations

import argparse
from pathlib import Path

from fem3d.demo_cases import run_beam_case, run_convergence_study


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
    result = run_beam_case(args.output, nx=args.nx, ny=args.ny, nz=args.nz)
    print(f"wrote {result.output}")
    print(f"max |u| = {result.max_displacement:.6e}")
    print(f"support reaction = {result.support_reaction}")


def _run_convergence(args: argparse.Namespace) -> None:
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
