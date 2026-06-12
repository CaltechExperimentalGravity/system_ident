"""Gaussian maximum-likelihood estimator (plant + noise model).

Wraps ``Optimal_Controls/GML_Estimator``. Drop-in alternative; build step 10.
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel
from .base import Estimator


class GMLEstimator(Estimator):
    """GML fit of plant (and noise model) — Optimal_Controls/GML_Estimator."""

    def fit(
        self,
        freq: np.ndarray,
        H_meas: np.ndarray,
        H_err: np.ndarray,
        model: TFModel,
    ) -> TFModel:
        raise NotImplementedError("GMLEstimator.fit lands in build step 10")
