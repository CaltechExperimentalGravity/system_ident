"""system_ident — real-time, optimal-excitation system identification for LIGO suspensions.

This package consolidates the proven research code in this repository (the
``sys_id_dev/sysIDlib.py`` engine, ``Plant_Model`` 3-DoF suspension, and the
``Optimal_Controls`` estimators) into a
single installable tool that runs iterative system identification of a 3-DoF
suspension (POS / Pitch / Yaw) against either real CDS hardware (awg inject +
nds2 readback) or a digital twin — through one identical channel API — with a
live operational dashboard and an operator STOP that performs a safe handoff.

See ``docs``/the plan for the full design. Heavy or optional dependencies
(matplotlib, control, the dashboard stack, and the CDS libraries) are imported
lazily by the submodules that need them, so ``import system_ident`` stays cheap.
"""

from __future__ import annotations

__version__ = "0.0.1"

# Lightweight public API (numpy-only). Heavier pieces (plant, dashboard, CDS
# backend) are imported from their submodules on demand.
from .model import TFModel
from .estimators.base import Estimator
from .design.base import InputDesigner
from .backends.base import ChannelBackend

__all__ = [
    "__version__",
    "TFModel",
    "Estimator",
    "InputDesigner",
    "ChannelBackend",
]
