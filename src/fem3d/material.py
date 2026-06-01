from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IsotropicMaterial:
    """Isotropic small-strain linear elastic material."""

    young: float
    poisson: float

    def __post_init__(self) -> None:
        if self.young <= 0.0:
            raise ValueError("young must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("poisson must satisfy -1 < poisson < 0.5")

    @property
    def lame_lambda(self) -> float:
        return self.young * self.poisson / (
            (1.0 + self.poisson) * (1.0 - 2.0 * self.poisson)
        )

    @property
    def shear_mu(self) -> float:
        return self.young / (2.0 * (1.0 + self.poisson))

    def elasticity_matrix(self) -> np.ndarray:
        lam = self.lame_lambda
        mu = self.shear_mu
        return np.array(
            [
                [lam + 2 * mu, lam, lam, 0, 0, 0],
                [lam, lam + 2 * mu, lam, 0, 0, 0],
                [lam, lam, lam + 2 * mu, 0, 0, 0],
                [0, 0, 0, mu, 0, 0],
                [0, 0, 0, 0, mu, 0],
                [0, 0, 0, 0, 0, mu],
            ],
            dtype=float,
        )
