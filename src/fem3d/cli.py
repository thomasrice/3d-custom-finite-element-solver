from __future__ import annotations

import argparse
from pathlib import Path

from fem3d.workflows import (
    format_beam_result,
    format_convergence_study,
    run_beam_case,
    run_convergence_study,
)


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
    print(format_beam_result(result))


def _run_convergence(args: argparse.Namespace) -> None:
    result = run_convergence_study(args.levels, vtk_dir=args.vtk_dir, csv=args.csv)
    print(format_convergence_study(result))


if __name__ == "__main__":
    main()
