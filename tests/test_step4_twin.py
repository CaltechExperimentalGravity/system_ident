"""Step-4: the digital-twin backend behind the channel API.

These are self-contained (no legacy oracle): they check that the twin applies
the plant transfer function to injected drive, returns reproducible sensor-noise
on quiet reads, ramps smoothly to zero, and routes channels per the config.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.signal as sig

from system_ident.backends.twin import TwinBackend
from system_ident.model import TFModel
from system_ident.plant import SuspensionPlant, double_pendulum

FS = 32.0
EXC = {"C1:EXC_POS": "POS"}
RB = {"C1:RESP_POS": "POS"}


def _single_dof_twin(sensor_asd=0.0, seed=0, ramp_s=0.0):
    # ramp_s=0 here: these are unit tests of the plant-filter / impulse / ramp_down
    # mechanics, where the actuator-safe injection ramp is orthogonal. The ramp default
    # itself is covered by test_inject_applies_soft_start_stop_ramp.
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    return TwinBackend(plant, EXC, RB, fs=FS, sensor_asd=sensor_asd, seed=seed, ramp_s=ramp_s)


def test_inject_applies_soft_start_stop_ramp():
    """By default the injected drive ramps on/off with a **3 s** Tukey taper at each
    end — actuator-safe; the middle stays at full amplitude and ramp_s=0 disables it."""
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    drive = np.ones(int(30 * FS))
    ramped = TwinBackend(plant, EXC, RB, fs=FS, ramp_s=3.0)
    ramped.inject("C1:EXC_POS", drive, FS)
    mon = ramped.read(["C1:EXC_POS"], duration=30.0)["C1:EXC_POS"]
    assert abs(mon[0]) < 1e-9 and abs(mon[-1]) < 1e-9        # tapered on and off
    assert mon[len(mon) // 2] == pytest.approx(1.0)          # full amplitude in the middle
    # the taper is specifically 3 s: full by 3 s in (and out), ~half-cosine at 1.5 s,
    # still ramping at 1 s — a 1 s ramp would already be at full amplitude here.
    assert mon[int(3.0 * FS)] == pytest.approx(1.0, abs=1e-6)
    assert mon[-int(3.0 * FS) - 1] == pytest.approx(1.0, abs=1e-6)
    assert mon[int(1.5 * FS)] == pytest.approx(0.5, abs=0.05)
    assert mon[int(1.0 * FS)] < 0.8                          # not yet full at 1 s (3 s ramp)

    plain = TwinBackend(plant, EXC, RB, fs=FS, ramp_s=0.0)
    plain.inject("C1:EXC_POS", drive, FS)
    assert plain.read(["C1:EXC_POS"], duration=30.0)["C1:EXC_POS"][0] == pytest.approx(1.0)


def test_inject_read_applies_plant_filter():
    twin = _single_dof_twin()
    rng = np.random.default_rng(1)
    drive = rng.standard_normal(int(8 * FS))
    twin.inject("C1:EXC_POS", drive, FS)

    out = twin.read(["C1:RESP_POS"], duration=8.0)["C1:RESP_POS"]

    b, a = sig.bilinear(double_pendulum().num, double_pendulum().den, FS)
    expected = sig.lfilter(b, a, drive)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=0)


def test_impulse_response_matches_continuous_plant_at_low_freq():
    # Long capture so the resonant ring-down (tau ~ Q/(pi f0) ~ 10 s) fully
    # decays and the FFT of the finite response approximates the DTFT.
    twin = _single_dof_twin()
    dur = 256.0
    n = int(dur * FS)
    impulse = np.zeros(n)
    impulse[0] = 1.0
    twin.inject("C1:EXC_POS", impulse, FS)
    resp = twin.read(["C1:RESP_POS"], duration=dur)["C1:RESP_POS"]

    H_twin = np.fft.rfft(resp)
    f = np.fft.rfftfreq(n, d=1 / FS)

    # Well below Nyquist (and off the sharp resonances) the bilinear-discretised
    # twin tracks the continuous plant model.
    band = (f > 0.05) & (f < 0.4)
    H_cont = double_pendulum().eval(f[band])
    np.testing.assert_allclose(H_twin[band], H_cont, rtol=0.05)


def test_quiet_read_is_reproducible_noise():
    asd = 1e-9
    a = _single_dof_twin(sensor_asd=asd, seed=42)
    b = _single_dof_twin(sensor_asd=asd, seed=42)
    ya = a.read(["C1:RESP_POS"], duration=64.0)["C1:RESP_POS"]
    yb = b.read(["C1:RESP_POS"], duration=64.0)["C1:RESP_POS"]

    np.testing.assert_array_equal(ya, yb)  # same seed -> same noise
    expected_std = asd * np.sqrt(FS / 2.0)
    assert np.std(ya) == pytest.approx(expected_std, rel=0.1)


def test_ramp_down_tapers_to_zero():
    twin = _single_dof_twin()
    drive = np.ones(int(10 * FS))
    twin.inject("C1:EXC_POS", drive, FS)
    twin.ramp_down("C1:EXC_POS", secs=2.0)

    monitored = twin.read(["C1:EXC_POS"], duration=10.0)["C1:EXC_POS"]
    n_ramp = int(2.0 * FS)
    assert monitored[0] == pytest.approx(1.0)          # starts at full amplitude
    assert abs(monitored[n_ramp - 1]) < 0.05           # tapered to ~0
    np.testing.assert_array_equal(monitored[n_ramp:], 0.0)  # silent thereafter


def test_inject_resamples_mismatched_rate():
    twin = _single_dof_twin()
    # drive supplied at 64 Hz, backend runs at 32 Hz -> resampled to 32*4 samples
    drive = np.zeros(int(4 * 64))
    twin.inject("C1:EXC_POS", drive, 64.0)
    out = twin.read(["C1:RESP_POS"], duration=4.0)["C1:RESP_POS"]
    assert out.shape == (int(4 * FS),)


def test_from_config_routes_channels_and_dofs_are_independent():
    spec = {
        "POS": {"resonances": [(0.6, 20.0)], "gain": 1.0},
        "PIT": {"resonances": [(2.0, 10.0)], "gain": 5.0},
    }
    plant = SuspensionPlant.from_resonance_spec(spec, fs=FS)
    config = {
        "channels": {
            "excitation": {"POS": "C1:EXC_1", "PIT": "C1:EXC_2"},
            "readback": {"POS": "C1:RESP_1", "PIT": "C1:RESP_2"},
        }
    }
    twin = TwinBackend.from_config(config, plant, fs=FS)
    assert twin.exc_channels == {"C1:EXC_1": "POS", "C1:EXC_2": "PIT"}

    # driving POS must not move PIT's readback
    drive = np.random.default_rng(0).standard_normal(int(4 * FS))
    twin.inject("C1:EXC_1", drive, FS)
    out = twin.read(["C1:RESP_1", "C1:RESP_2"], duration=4.0)
    assert np.any(out["C1:RESP_1"] != 0)
    np.testing.assert_array_equal(out["C1:RESP_2"], 0.0)


def test_unknown_channel_raises():
    twin = _single_dof_twin()
    with pytest.raises(KeyError):
        twin.inject("C1:NOPE", np.zeros(10), FS)
    with pytest.raises(KeyError):
        twin.read(["C1:NOPE"], duration=1.0)


def test_disturbance_asd_colors_quiet_readout():
    """Input-referred disturbance noise should colour the quiet readout through
    the plant (plant-shaped PSD), while sensor noise alone gives a flat PSD."""
    plant = SuspensionPlant({"POS": double_pendulum()}, fs=FS)
    dur = 512.0
    nperseg = int(64 * FS)  # freq resolution ~0.016 Hz, resolves Q=30 resonance

    # disturbance > 0, sensor = 0: PSD should be plant-shaped (high peak/median)
    twin_dist = TwinBackend(
        plant, EXC, RB, fs=FS, disturbance_asd=1e-6, sensor_asd=0.0, seed=0
    )
    y_dist = twin_dist.read(["C1:RESP_POS"], duration=dur)["C1:RESP_POS"]
    f, Pxx_dist = sig.welch(y_dist, fs=FS, nperseg=nperseg)
    pos = f > 0
    ratio_dist = Pxx_dist[pos].max() / np.median(Pxx_dist[pos])
    assert ratio_dist > 20.0, (
        f"expected plant-shaped PSD (peak/median > 20) with disturbance_asd, "
        f"got {ratio_dist:.1f}"
    )

    # sensor_asd > 0, disturbance = 0: PSD should be approximately flat
    twin_sensor = TwinBackend(
        plant, EXC, RB, fs=FS, disturbance_asd=0.0, sensor_asd=1e-6, seed=1
    )
    y_sensor = twin_sensor.read(["C1:RESP_POS"], duration=dur)["C1:RESP_POS"]
    _, Pxx_sensor = sig.welch(y_sensor, fs=FS, nperseg=nperseg)
    ratio_sensor = Pxx_sensor[pos].max() / np.median(Pxx_sensor[pos])
    assert ratio_sensor < 5.0, (
        f"expected flat PSD (peak/median < 5) for white sensor noise, "
        f"got {ratio_sensor:.1f}"
    )
