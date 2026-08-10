"""Step-8: config loading/validation and the CLI run path (twin)."""

from __future__ import annotations

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
def test_cli_twin_run_succeeds(capsys):
    assert main(["run", str(DEMO), "--twin", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "DONE" in out
    assert "POS" in out and "PIT" in out


def test_cli_requires_a_backend_choice(capsys):
    # --cds now exists (Stage F), so this is "choose one", not "CDS isn't available".
    assert main(["run", str(DEMO)]) == 2
    assert "choose a backend" in capsys.readouterr().err


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

    assert main(["run", str(p), "--twin", "--yes"]) == 1
    out = capsys.readouterr().out
    assert "ABORTED" in out and "actuator saturation" in out
    assert "safe-state handoff completed" in out
