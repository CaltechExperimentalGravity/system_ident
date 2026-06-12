"""Estimator strategies — pluggable ways to fit a TFModel from measured data.

This is one of the two genuinely-swappable axes in the design (the other is
:mod:`ligo_sysid.design`). Concrete strategies wrap existing repo code behind
the :class:`~ligo_sysid.estimators.base.Estimator` interface.
"""

from .base import Estimator

__all__ = ["Estimator"]
