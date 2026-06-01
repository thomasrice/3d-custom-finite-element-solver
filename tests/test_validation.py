import numpy as np

from fem3d.validation import ConvergenceRow, convergence_rates, write_convergence_csv


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
