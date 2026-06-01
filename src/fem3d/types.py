from __future__ import annotations

from typing import Callable

import numpy as np

VectorFunction = Callable[[np.ndarray], np.ndarray]
VectorValue = VectorFunction | np.ndarray
