"""Stage F: wiring -- ``--cds``, ``config.py::build_cds_backend``, the
``BACKENDS`` registry, and the site-agnostic example config.
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import scipy.signal as sig
import yaml

import system_ident
from system_ident.cli import main
from system_ident.config import BACKENDS, ConfigError, RunConfig

from _fake_cds import FakeArbitraryLoop, FakeArbitraryStream, install

EXAMPLE = Path(system_ident.__file__).parent / "configs" / "cds_twin_transport.yml"
DEMO = Path(system_ident.__file__).parent / "configs" / "twin_demo.yml"


@pytest.fixture(autouse=True)
def _reset():
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()
    os.environ.setdefault("IFO", "X1")
    yield
    FakeArbitraryLoop.reset()
    FakeArbitraryStream.reset()
    sys.modules.pop("fake_x1hsts", None)


def test_backends_registry_matches_the_flags():
    assert set(BACKENDS) == {"twin", "rtsfreerun", "cds"}
    assert all(hasattr(RunConfig, m) for m in BACKENDS.values())


def test_example_config_loads_and_is_site_agnostic():
    cfg = RunConfig.load(EXAMPLE)
    assert cfg.raw["cds"]["transport"] == "twin"
    text = EXAMPLE.read_text()
    # no site IFO value or a real 40m-style channel prefix hardcoded
    assert "site_ifo_env: IFO" not in text.replace("# site_ifo_env: IFO", "")
    for placeholder in ("X1:", "H1:", "L1:"):
        assert placeholder not in text


def _install_fake_x1hsts():
    """A minimal stand-in for the compiled ``x1hsts`` model, registered under
    a name the example config doesn't use, so tests never depend on the real
    twin box being built."""
    mod = types.ModuleType("fake_x1hsts")

    class _Model:
        sample_rate = 16384.0

        def __init__(self):
            plant = system_ident.TFModel.from_resonances([(0.67, 50)], 0.3)
            self._b, self._a = sig.bilinear(plant.num, plant.den, self.sample_rate)
            self._zi = np.zeros(max(len(self._a), len(self._b)) - 1)
            self._pending = None
            self._captured = []
            self._rng = np.random.default_rng(0)

        def fetch_later(self, t0, t1, names):
            want = int(round((t1 - t0) * self.sample_rate))
            self._pending = (list(names), want)
            return lambda: self._captured

        def run(self, cycles, excitations=None, excitation_data=None):
            n = int(cycles)
            drive = np.zeros(n)
            if excitations and excitation_data is not None:
                ed = np.asarray(excitation_data, dtype=float)
                for j, ch in enumerate(excitations):
                    if ch == "COIL_DRIVER_EXC":
                        drive = drive + (ed[:, j] if ed.ndim == 2 else ed)[:n]
            y, self._zi = sig.lfilter(self._b, self._a, drive, zi=self._zi)
            y = y + self._rng.standard_normal(n) * 1e-9   # a quiet-time noise floor -> Pyy != 0
            probes = {"READOUT_NOISE_OUT": y, "COIL_DRIVER_OUT": drive}
            if self._pending is not None:
                names, want = self._pending
                buf = types.SimpleNamespace
                self._captured = [buf(data=probes.get(nm, np.zeros(n))[:want]) for nm in names]
                self._pending = None

    mod.fake_x1hsts = _Model
    sys.modules["fake_x1hsts"] = mod


def test_build_cds_backend_twin_transport_end_to_end():
    _install_fake_x1hsts()
    raw = yaml.safe_load(EXAMPLE.read_text())
    del raw["rtsfreerun"]["scenario"]      # the real twin-checkout scenario file isn't present here
    raw["rtsfreerun"]["model"] = "fake_x1hsts"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "cds_twin_fake.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        cfg = RunConfig.load(cfg_path)
        backend = cfg.build_cds_backend()
        assert backend.exc_channels == {"COIL_DRIVER_EXC": "POS"}
        watchdog = cfg.build_watchdog(backend)
        assert watchdog.exc_channels == backend.exc_channels


def test_build_cds_backend_requires_drive_channel():
    raw = yaml.safe_load(EXAMPLE.read_text())
    del raw["channels"]["drive"]
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "no_drive.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        cfg = RunConfig.load(cfg_path)
        with pytest.raises(ConfigError, match="channels.drive"):
            cfg.build_cds_backend()


def test_build_cds_backend_awg_nds_via_fakes():
    from _fake_cds import FakeCDSWorld
    world = FakeCDSWorld()
    plant = system_ident.TFModel.from_resonances([(0.67, 50)], 0.3)
    b, a = sig.bilinear(plant.num, plant.den, 16384.0)
    world.getdata.add_exc_channel("COIL_DRIVER_EXC", rate=16384.0, testpoint=True)
    world.getdata.add_readback_channel("COIL_DRIVER_OUT", exc_channel="COIL_DRIVER_EXC",
                                       plant=([1.0], [1.0]), rate=16384.0, testpoint=True)
    world.getdata.add_readback_channel("READOUT_NOISE_OUT", exc_channel="COIL_DRIVER_EXC",
                                       plant=(b, a), rate=16384.0, testpoint=True)
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["cds"]["transport"] = "awg_nds"
    import tempfile
    with install(world):
        with tempfile.TemporaryDirectory() as d:
            cfg_path = Path(d) / "cds_awg.yml"
            cfg_path.write_text(yaml.safe_dump(raw))
            cfg = RunConfig.load(cfg_path)
            backend = cfg.build_cds_backend()
            assert backend.exc_channels == {"COIL_DRIVER_EXC": "POS"}


def test_invalid_transport_kind_is_a_config_error():
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["cds"]["transport"] = "carrier_pigeon"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "bad_transport.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        cfg = RunConfig.load(cfg_path)
        with pytest.raises(ConfigError, match="cds.transport"):
            cfg.build_cds_backend()


def test_freq_max_above_08_nyquist_warns():
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["measurement"]["freq_max"] = 120.0   # > 0.8 * 256/2 = 102.4
    del raw["rtsfreerun"]["scenario"]
    raw["rtsfreerun"]["model"] = "fake_x1hsts"
    _install_fake_x1hsts()
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "hi_freq.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        cfg = RunConfig.load(cfg_path)
        with pytest.warns(UserWarning, match="0.8x Nyquist"):
            cfg.build_cds_backend()


def test_cli_yes_rejected_for_cds_real_hardware(capsys):
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["cds"]["transport"] = "awg_nds"
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "awg.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        assert main(["run", str(cfg_path), "--cds", "--yes"]) == 2
        assert "--yes cannot be used with --cds" in capsys.readouterr().err


def test_cli_yes_allowed_for_cds_twin_transport(capsys):
    # cds.transport: twin needs no per-injection approval at all (no live
    # hardware) -- --yes only skips the CLI-level pre-flight confirm, and a
    # full campaign should actually complete end-to-end.
    _install_fake_x1hsts()
    raw = yaml.safe_load(EXAMPLE.read_text())
    del raw["rtsfreerun"]["scenario"]
    raw["rtsfreerun"]["model"] = "fake_x1hsts"
    raw["stop_criteria"]["max_iter"] = 1
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        cfg_path = Path(d) / "twin.yml"
        cfg_path.write_text(yaml.safe_dump(raw))
        code = main(["run", str(cfg_path), "--cds", "--yes"])
        out = capsys.readouterr().out
        assert code == 0, out
        assert "running cds campaign" in out
