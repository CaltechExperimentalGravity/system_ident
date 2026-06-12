"""Weighted least-squares estimator.

Wraps ``Optimal_Controls/Least_Squares``. Drop-in alternative; build step 10.
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel
from .base import Estimator


class WeightedLSEstimator(Estimator):
    """Frequency-domain weighted LS fit (Optimal_Controls/Least_Squares)."""

    def fit(
        self,
        freq: np.ndarray,
        H_meas: np.ndarray,
        H_err: np.ndarray,
        model: TFModel,
    ) -> TFModel:
        raise NotImplementedError("WeightedLSEstimator.fit lands in build step 10")
