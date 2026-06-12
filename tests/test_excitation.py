"""Tests for excitation.timeseries_from_asd, including the Tukey-window
t_ramp soft start/stop feature."""

from __future__ import annotations

import numpy as np
import pytest

from system_ident.excitation import timeseries_from_asd

FS = 32.0
DURATION = 10.0
FREQ = np.array([0.1, 1.0, 5.0, 10.0, 16.0])
ASD = np.ones(len(FREQ))  # flat spectrum for simplicity
T_RAMP = 1.0  # 1 second ramp on each side


def test_tukey_t_ramp_zero_equals_no_ramp():
    """t_ramp=0 (the default) must produce bit-identical output to omitting t_ramp."""
    data_default = timeseries_from_asd(DURATION, FS, FREQ, ASD, seed=0)
    data_explicit_zero = timeseries_from_asd(DURATION, FS, FREQ, ASD, seed=0, t_ramp=0.0)
    np.testing.assert_array_equal(data_explicit_zero, data_default)


def test_tukey_endpoints_are_tapered():
    """With t_ramp > 0 the first and last samples must be ~0 (tapered)."""
    data = timeseries_from_asd(DURATION, FS, FREQ, ASD, seed=0, t_ramp=T_RAMP)
    assert abs(data[0]) < 1e-12, f"first sample not tapered to 0: {data[0]}"
    assert abs(data[-1]) < 1e-12, f"last sample not tapered to 0: {data[-1]}"


def test_tukey_central_region_untapered():
    """The central flat region must equal the no-ramp output (window = 1 there)."""
    data_no_ramp = timeseries_from_asd(DURATION, FS, FREQ, ASD, seed=0)
    data_ramp = timeseries_from_asd(DURATION, FS, FREQ, ASD, seed=0, t_ramp=T_RAMP)
    N = int(DURATION * FS)
    # taper spans int(T_RAMP * FS) samples on each side
    n_taper = int(T_RAMP * FS)
    np.testing.assert_allclose(
        data_ramp[n_taper : N - n_taper],
        data_no_ramp[n_taper : N - n_taper],
        err_msg="central (flat) region of Tukey window should leave data unchanged",
    )
