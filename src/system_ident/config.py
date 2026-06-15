"""Run configuration: load + validate the YAML that drives a run, and build the
components it specifies.

A single YAML config is the reproducible base for a run (suspension, channels,
estimator/designer choices, safety limits, excitation mode, stop criteria); CLI
flags override it. Loading uses ``yaml.safe_load`` (matching the repo
convention) and validates required sections, failing fast on bad input.

The ``build_*`` helpers turn a validated config into the concrete objects the
:class:`~system_ident.loop.SysIDLoop` needs, keeping that wiring (and the
strategy-name registries) in one place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from .backends.twin import TwinBackend
from .design.pintelon import PintelonSchoukensDesigner
from .estimators.gml import GMLEstimator
from .model import TFModel
from .plant import SuspensionPlant
from .safety import SafetyLimits, Watchdog

# Strategy-name -> implementation. Only the Pintelon-Schoukens maximum-
# likelihood estimator ("gml"/"ml") is wired; any other choice fails loudly
# instead of silently doing the wrong thing.
ESTIMATORS = {
    "gml": GMLEstimator,
    "ml": GMLEstimator,
}
DESIGNERS = {"pintelon_schoukens": PintelonSchoukensDesigner}

REQUIRED = {
    "run": [],
    "channels": ["excitation", "readback"],
    "measurement": ["fs", "freq_min", "freq_max", "segment_duration", "px_total"],
    "strategy": ["estimator", "input_designer"],
    "safety": ["actuator_sat", "rms_ceiling"],
    "stop_criteria": ["uncertainty_target"],
}


class ConfigError(ValueError):
    """Raised when a run config is missing or malformed."""


def _parse_controllers(spec: dict) -> dict:
    """Normalise a ``twin.controllers`` section to ``{dof: (num, den)}``.

    Each controller is a continuous-time ``C(s) = num/den`` given either as
    ``{num: [...], den: [...]}`` or as a ``[num, den]`` pair.
    """
    out: dict[str, tuple] = {}
    for dof, c in spec.items():
        if isinstance(c, dict):
            out[dof] = ([float(x) for x in c["num"]], [float(x) for x in c["den"]])
        else:
            num, den = c
            out[dof] = ([float(x) for x in num], [float(x) for x in den])
    return out


def _resonance_spec(section: dict) -> dict:
    """Normalise a ``{dof: {resonances: [[f0, Q], ...], gain: k}}`` mapping."""
    return {
        dof: {
            "resonances": [tuple(r) for r in d["resonances"]],
            "gain": float(d["gain"]),
            "zeros": [complex(z) for z in d.get("zeros", [])],
        }
        for dof, d in section.items()
    }


@dataclass
class RunConfig:
    """Validated configuration for one sysID run."""

    raw: dict
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "RunConfig":
        """Load and validate a run config from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"config file not found: {path}")
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ConfigError(f"config must be a mapping, got {type(raw).__name__}")
        cls._validate(raw)
        return cls(raw=raw, path=path)

    @staticmethod
    def _validate(raw: dict) -> None:
        for section, keys in REQUIRED.items():
            if section not in raw:
                raise ConfigError(f"missing required section: {section!r}")
            for key in keys:
                if key not in raw[section]:
                    raise ConfigError(f"missing required key: {section}.{key}")
        est = raw["strategy"]["estimator"]
        des = raw["strategy"]["input_designer"]
        if est not in ESTIMATORS:
            raise ConfigError(
                f"estimator {est!r} not available; choose from {sorted(ESTIMATORS)}"
            )
        if des not in DESIGNERS:
            raise ConfigError(
                f"input_designer {des!r} not available; choose from {sorted(DESIGNERS)}"
            )

    # -- CLI flag overrides --------------------------------------------------
    def apply_overrides(
        self,
        estimator: str | None = None,
        segment_duration: float | None = None,
        px_total: float | None = None,
    ) -> None:
        """Apply CLI flag overrides in place, re-validating afterwards."""
        if estimator is not None:
            self.raw["strategy"]["estimator"] = estimator
        if segment_duration is not None:
            self.raw["measurement"]["segment_duration"] = float(segment_duration)
        if px_total is not None:
            self.raw["measurement"]["px_total"] = float(px_total)
        self._validate(self.raw)

    # -- component factories -------------------------------------------------
    @property
    def fs(self) -> float:
        return float(self.raw["measurement"]["fs"])

    def build_plant(self) -> SuspensionPlant:
        if "twin" not in self.raw or "plant" not in self.raw["twin"]:
            raise ConfigError("twin runs need a 'twin.plant' resonance spec")
        spec = _resonance_spec(self.raw["twin"]["plant"])
        return SuspensionPlant.from_resonance_spec(spec, self.fs)

    def build_twin_backend(self, seed: int | None = None) -> TwinBackend:
        twin = self.raw.get("twin", {})
        sensor_asd = float(twin.get("sensor_asd", 0.0))
        disturbance_asd = float(twin.get("disturbance_asd", 0.0))
        # Optional closed-loop: per-DoF controllers C(s) and where the excitation
        # is injected relative to the controller. The reference-based FRF cancels
        # the controller, so the loop still recovers the open-loop plant.
        extra: dict = {}
        if "controllers" in twin:
            extra["controllers"] = _parse_controllers(twin["controllers"])
        if "injection_point" in twin:
            extra["injection_point"] = twin["injection_point"]
        return TwinBackend.from_config(
            self.raw, self.build_plant(), fs=self.fs,
            sensor_asd=sensor_asd, disturbance_asd=disturbance_asd, seed=seed,
            **extra,
        )

    def build_rtsfreerun_backend(self, seed: int | None = None):
        """Build the RTSfreerun (digital-twin) backend from a ``rtsfreerun`` section.

        Drives a compiled rtsfreerun CDS model (``GIT/digital_twin/``) under its own
        realistic noise. The model module is lazy-imported, so this only succeeds
        in an environment where the named model is installed.
        """
        from .backends.rtsfreerun_adapter import RTSfreerunBackend
        if "rtsfreerun" not in self.raw or "model" not in self.raw["rtsfreerun"]:
            raise ConfigError("rtsfreerun runs need a 'rtsfreerun.model' name")
        return RTSfreerunBackend.from_config(self.raw, fs=self.fs, seed=seed)

    def build_priors(self) -> dict:
        """Return per-DoF prior :class:`TFModel` models (one per DoF)."""
        if "priors" not in self.raw:
            raise ConfigError("runs need a 'priors' section (one model per DoF)")
        spec = _resonance_spec(self.raw["priors"])
        return {
            dof: TFModel.from_resonances(d["resonances"], d["gain"], zeros=d["zeros"])
            for dof, d in spec.items()
        }

    def build_estimator(self):
        return ESTIMATORS[self.raw["strategy"]["estimator"]]()

    def build_designer(self):
        return DESIGNERS[self.raw["strategy"]["input_designer"]]()

    def build_watchdog(self, backend) -> Watchdog:
        return Watchdog(backend, SafetyLimits.from_config(self.raw))
