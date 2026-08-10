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

import warnings
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

# backend-flag name -> the RunConfig.build_*_backend method that constructs
# it. Mirrors ESTIMATORS/DESIGNERS; kept as method NAMES (not classes/no-arg
# callables) because backend construction needs real per-backend wiring
# (transport selection, model lazy-import, seed) that a bare registry can't
# express uniformly -- see each build_*_backend for that wiring.
BACKENDS = {
    "twin": "build_twin_backend",
    "rtsfreerun": "build_rtsfreerun_backend",
    "cds": "build_cds_backend",
}

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
        # loop.py's adaptive transient guard (_choose_transient) needs headroom to
        # adapt at all -- below this margin it can drop every period but one,
        # leaving _estimate_tf_periodic with P_eff < 2 (a shipped-config-reachable
        # case: configs/rtsfreerun_hsts.yml's own n_transient recommendation hits
        # exactly this). P_eff < 2 is now excluded safely (#6), but a measurement
        # that silently contributes nothing is still a config mistake worth
        # catching before any hardware time is spent on it.
        n_seg = int(raw["measurement"].get("n_segments", 8))
        n_transient = int(raw["measurement"].get("n_transient", 1))
        if n_seg < n_transient + 3:
            raise ConfigError(
                f"measurement.n_segments ({n_seg}) must be >= "
                f"measurement.n_transient ({n_transient}) + 3"
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

    def build_cds_backend(self, seed: int | None = None):
        """Build ``CDSBackend`` from the ``cds`` section: ``transport: awg_nds
        | twin`` selects real hardware or a compiled-model twin transport,
        both behind the identical backend/channel API (spec S4).

        ``seed`` is accepted for symmetry with the other ``build_*_backend``
        methods (the CLI calls whichever one uniformly via ``BACKENDS``) but
        unused here -- a hardware/twin-transport run has no RNG of its own.
        """
        from .backends.cds import CDSBackend
        from .backends.cds_transport import AWGNDSTransport, TwinTransport

        ch = self.raw["channels"]
        # channels.drive mandatory (spec S2.1 item 3): the fallback to the
        # excitation channel itself (what loop.py:90 does for every OTHER
        # backend) returns the stashed drive on hardware, not a real
        # readback -- the exact 200%-error configuration S2.1 exists to rule
        # out. A ConfigError here, not a warning, is deliberate.
        if not ch.get("drive"):
            raise ConfigError(
                "the cds backend requires channels.drive for every DoF -- falling back "
                "to the excitation channel returns the stashed drive, not a real "
                "readback (spec S2.1); this is the one backend where that fallback is "
                "not just imprecise but a 200%-error configuration"
            )

        m = self.raw["measurement"]
        # CDSBackend.read() requires integer-second record durations (NDS
        # fetches are second-granular); catch a fractional total here, at
        # config time, instead of as a confusing TransportUnavailable at the
        # first quiet read. CDS-specific -- the twin backends accept any
        # duration, so this does not belong in _validate.
        total_dur = float(m["segment_duration"]) * int(m.get("n_segments", 8))
        if abs(total_dur - round(total_dur)) > 1e-9:
            raise ConfigError(
                f"the cds backend needs segment_duration * n_segments to be a whole "
                f"number of seconds, got {m['segment_duration']} * "
                f"{m.get('n_segments', 8)} = {total_dur}"
            )
        freq_max = float(m["freq_max"])
        fs = self.fs
        if freq_max > 0.8 * (fs / 2):
            warnings.warn(
                f"measurement.freq_max ({freq_max} Hz) is above 0.8x Nyquist "
                f"({0.8 * fs / 2:.3g} Hz of {fs} Hz fs): the decimation anti-alias "
                "filter's rolloff there costs SNR, not bias -- it is LTI and applied "
                "to both X and Y, so it cancels exactly in Ybar/Xbar, but ONLY "
                "because X is a real readback (spec S2.1, S7); it does not cancel if "
                "X were ever synthesised.",
                stacklevel=2,
            )

        cds_cfg = self.raw.get("cds", {})
        transport_kind = cds_cfg.get("transport", "twin")
        # Spec S1: "simulation / twin / smoke tests need no such approval."
        # CDSBackend.inject()'s approval gate defaults to a real, interactive,
        # deny-by-default prompt (correct for awg_nds); a twin transport has
        # no live hardware behind it, so auto-approve instead of prompting
        # for something that can never actuate anything real.
        authorizer = (lambda prompt: True) if transport_kind == "twin" else None
        if transport_kind == "awg_nds":
            transport = AWGNDSTransport(site_ifo_env=cds_cfg.get("site_ifo_env", "IFO"))
        elif transport_kind == "twin":
            if "rtsfreerun" not in self.raw or "model" not in self.raw["rtsfreerun"]:
                raise ConfigError(
                    "cds.transport: twin needs a 'rtsfreerun.model' name (it drives the "
                    "same compiled model TwinTransport wraps)"
                )
            import importlib
            rts = self.raw["rtsfreerun"]
            model_name = rts["model"]
            pkg = importlib.import_module(model_name)
            mdl = getattr(pkg, model_name)()
            if "scenario" in rts:
                # Same plant-loading step RTSfreerunBackend does (foton ZPK ->
                # cdsFilt modules) -- without it the bare model runs whatever
                # default dynamics are baked into the class, not the scenario
                # this config's priors were written against.
                from .backends.rtsfreerun_oracle import apply_scenario_init, load_scenario
                scen = rts["scenario"]
                apply_scenario_init(mdl, scen if isinstance(scen, dict) else load_scenario(scen))
            transport = TwinTransport(mdl)
        else:
            raise ConfigError(
                f"cds.transport must be 'awg_nds' or 'twin', got {transport_kind!r}"
            )

        return CDSBackend.from_config(self.raw, transport=transport, authorizer=authorizer)

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
