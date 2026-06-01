import numpy as np

from fem3d.cli import main
from fem3d.validation import ConvergenceRow, convergence_rates, write_convergence_csv
from fem3d.workflows import run_beam_case, run_convergence_study


def test_convergence_rates_and_csv_report(tmp_path):
    rows = [
        ConvergenceRow(h=0.5, dofs=81, l2=0.25, h1_seminorm=0.5),
        ConvergenceRow(h=0.25, dofs=375, l2=0.0625, h1_seminorm=0.25),
        ConvergenceRow(h=0.125, dofs=2187, l2=0.015625, h1_seminorm=0.125),
    ]

    l2_rate, h1_rate = convergence_rates(rows)
    path = tmp_path / "convergence.csv"
    write_convergence_csv(path, rows)

    assert np.isclose(l2_rate, 2.0)
    assert np.isclose(h1_rate, 1.0)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "h,dofs,l2,h1_seminorm",
        "0.5,81,2.5000000000000000e-01,5.0000000000000000e-01",
        "0.25,375,6.2500000000000000e-02,2.5000000000000000e-01",
        "0.125,2187,1.5625000000000000e-02,1.2500000000000000e-01",
    ]


def test_cli_writes_convergence_csv(tmp_path, capsys):
    path = tmp_path / "convergence.csv"

    main(["convergence", "--levels", "2", "4", "--csv", str(path)])

    assert path.exists()
    assert "L2 rate" in capsys.readouterr().out


def test_workflows_write_expected_artifacts(tmp_path):
    beam = run_beam_case(tmp_path / "beam.vtk", nx=2, ny=1, nz=1)
    convergence = run_convergence_study([2, 4], csv=tmp_path / "convergence.csv")

    assert beam.output.exists()
    assert beam.max_displacement > 0.0
    assert np.isclose(beam.support_reaction[2], 1.0)
    assert convergence.csv is not None and convergence.csv.exists()
    assert convergence.l2_rate is not None and convergence.l2_rate > 1.8
