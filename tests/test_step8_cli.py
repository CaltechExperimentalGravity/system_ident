"""Step-8: config loading/validation and the CLI run path (twin)."""

from __future__ import annotations

import importlib.util
import socket
from pathlib import Path

import pytest
import yaml

import system_ident
from system_ident.cli import main
from system_ident.config import ConfigError, RunConfig

DEMO = Path(system_ident.__file__).parent / "configs" / "twin_demo.yml"


# --- config -------------------------------------------------------------------
def test_demo_config_loads_and_builds():
    cfg = RunConfig.load(DEMO)
    plant = cfg.build_plant()
    priors = cfg.build_priors()
    assert set(plant.dofs) == {"POS", "PIT"}
    assert set(priors) == {"POS", "PIT"}
    backend = cfg.build_twin_backend(seed=0)
    assert backend.exc_channels  # channel maps populated
    assert cfg.build_estimator() and cfg.build_designer()


def test_missing_section_rejected(tmp_path):
    bad = tmp_path / "bad.yml"
    bad.write_text("run: {}\nchannels: {excitation: {}, readback: {}}\n")
    with pytest.raises(ConfigError, match="measurement"):
        RunConfig.load(bad)


def test_unknown_strategy_rejected(tmp_path):
    raw = yaml.safe_load(DEMO.read_text())
    raw["strategy"]["estimator"] = "nope"
    p = tmp_path / "c.yml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ConfigError, match="estimator 'nope' not available"):
        RunConfig.load(p)


def test_apply_overrides():
    cfg = RunConfig.load(DEMO)
    cfg.apply_overrides(segment_duration=32.0, px_total=2.0)
    assert cfg.raw["measurement"]["segment_duration"] == 32.0
    assert cfg.raw["measurement"]["px_total"] == 2.0
    with pytest.raises(ConfigError):
        cfg.apply_overrides(estimator="bogus")


# --- CLI ----------------------------------------------------------------------
# NB: the run tests pass --no-dashboard. Without it, a CLI run in an env that has
# the dashboard extra opens a real listening socket on the developer's machine and
# leaves the daemon serve() thread alive for the rest of the session.
def test_cli_twin_run_succeeds(capsys):
    assert main(["run", str(DEMO), "--twin", "--yes", "--no-dashboard"]) == 0
    out = capsys.readouterr().out
    assert "DONE" in out
    assert "POS" in out and "PIT" in out


def test_cli_requires_twin(capsys):
    assert main(["run", str(DEMO)]) == 2
    assert "CDS-hardware backend is not available" in capsys.readouterr().err


def test_cli_missing_file(capsys):
    assert main(["run", "/no/such/config.yml", "--twin"]) == 2
    assert "config file not found" in capsys.readouterr().err


def test_cli_confirm_declined(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "no")
    assert main(["run", str(DEMO), "--twin"]) == 1
    assert "aborted before injection" in capsys.readouterr().out


def test_cli_abort_on_saturation(tmp_path, capsys):
    raw = yaml.safe_load(DEMO.read_text())
    raw["safety"]["actuator_sat"] = 1.0e-12  # any drive breaches
    p = tmp_path / "sat.yml"
    p.write_text(yaml.safe_dump(raw))

    assert main(["run", str(p), "--twin", "--yes", "--no-dashboard"]) == 1
    out = capsys.readouterr().out
    assert "ABORTED" in out and "actuator saturation" in out
    assert "safe-state handoff completed" in out


# --- dashboard startup --------------------------------------------------------
# Both need a server to actually start, so they only apply where the extra is
# installed; without it the CLI reports "dashboard extra not installed" instead.
HAVE_DASHBOARD = bool(
    importlib.util.find_spec("fastapi") and importlib.util.find_spec("uvicorn")
)


@pytest.mark.skipif(not HAVE_DASHBOARD, reason="dashboard extra not installed")
def test_cli_reports_unavailable_dashboard_port_and_runs_headless(capsys):
    """A taken port must be diagnosed in the caller's thread.

    Binding inside the daemon serve() thread makes uvicorn's failure invisible:
    the CLI prints a dashboard URL, the operator gets no dashboard, and nothing
    says why. The run itself must still complete.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
        taken.bind(("127.0.0.1", 0))
        taken.listen(1)
        port = taken.getsockname()[1]
        assert main(["run", str(DEMO), "--twin", "--yes", "--port", str(port)]) == 0

    out = capsys.readouterr().out
    assert "DONE" in out                                   # run unaffected
    assert f"port {port}" in out and "headless" in out      # and diagnosed
    assert f"http://127.0.0.1:{port}" not in out            # no dead URL offered


@pytest.mark.skipif(not HAVE_DASHBOARD, reason="dashboard extra not installed")
def test_cli_dashboard_reports_the_port_it_actually_bound(capsys):
    """--port 0 lets the OS choose; the operator must be told the real port."""
    assert main(["run", str(DEMO), "--twin", "--yes", "--port", "0"]) == 0
    out = capsys.readouterr().out
    assert "http://127.0.0.1:0" not in out
    assert "dashboard: http://127.0.0.1:" in out
