"""Vector-fitting estimator.

Wraps ``QuadPend/vectfit2.py`` (Gustavsen-Semlyen rational approximation).
Drop-in alternative; build step 10.
"""

from __future__ import annotations

import numpy as np

from ..model import TFModel
from .base import Estimator


class VectfitEstimator(Estimator):
    """Rational approximation via vector fitting (QuadPend/vectfit2)."""

    def fit(
        self,
        freq: np.ndarray,
        H_meas: np.ndarray,
        H_err: np.ndarray,
        model: TFModel,
    ) -> TFModel:
        raise NotImplementedError("VectfitEstimator.fit lands in build step 10")
