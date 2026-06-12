"""Shared test fixtures.

The legacy ``sys_id_dev/sysIDlib.py`` engine is loaded as an oracle to validate
the ported package against. It targets old numpy/scipy, so we (a) stub ``h5py``
(imported at module top but unused by the functions we call) and (b) restore the
``np.int`` / ``scipy.integrate.trapz`` aliases that newer releases removed. Only
those names are shimmed — the engine's math runs untouched.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.integrate as _integ


def load_oracle():
    """Import the legacy ``sysIDlib`` with minimal compatibility shims."""
    path = Path(__file__).resolve().parents[1] / "sys_id_dev" / "sysIDlib.py"
    if not path.exists():
        pytest.skip(f"legacy validation reference not present: {path}")
    sys.modules.setdefault("h5py", types.ModuleType("h5py"))
    if not hasattr(np, "int"):
        np.int = int  # alias removed in numpy 2.x; legacy default-logflag path needs it
    if not hasattr(_integ, "trapz"):
        _integ.trapz = _integ.trapezoid  # renamed in scipy 1.14+
    spec = importlib.util.spec_from_file_location("sysIDlib_legacy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def oracle():
    return load_oracle()
