"""Stage 1 — 6-DOF MIMO Pintelon–Schoukens recovery of the 40m SOS, in-process.

The pyctl rehearsal / signoff gate for the SOS sysID campaign: drive the analytic SOS plant
(:mod:`system_ident.sos_plant`, loaded as ``sos_6dof``) with periodic multisines, recover its
six modes with the shared machinery (:mod:`system_ident.mimo_campaign`,
:mod:`system_ident.mimo_fit`), and check that the recovery agrees with the plant's own
eigenmodes **within the Cramér–Rao bound**. Runs anywhere — no compiled twin.

Design, and why it is two focused campaigns rather than one broadband sweep
---------------------------------------------------------------------------
The SOS modes span 0.6 → 22.6 Hz (nearly two decades) at Q≈50, so no single record both
resolves the 0.6 Hz linewidth (Γ = f0/Q = 0.012 Hz ⇒ df ≲ 2 mHz) and reaches 22.6 Hz
efficiently. A single sweep spread that thin also collapses the per-bin campaign covariance
to rank 1 off-resonance (the sensor-noise floor set from a broadband ``median|G|`` underflows
where the plant response is tiny), which makes the 2×2 {L,P} whitening singular. Two focused
campaigns, each with lines clustered on their modes (Fisher-optimal per ``CLAUDE.md``) and an
**absolute** on-resonance noise floor, both recover within the CRB:

* **Low band** (0.4–1.2 Hz) drives L, P, T, Y. L (1.000 Hz) and T (0.998 Hz) are a *spatial
  doublet* — near-coincident in frequency, orthogonal in DOF — so a single shared-pole MIMO
  fit would collapse them. ``fit_block_decoupled`` separates them: the coupled {L,P} 2×2
  block never drives T, and T is fit alone. Y is an independent SISO mode.
* **High band** (14–26 Hz) drives V, R — well-separated diagonal SISO modes.

The block structure follows the plant physics: only L–P is coupled (break-off torque);
T, V, R, Y are diagonal. See :func:`system_ident.sos_plant.stiffness_matrix`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .reduced_plant import ReducedStateSpacePlant
from .backends.reduced import ReducedPlantBackend
from .mimo_campaign import assemble_campaign
from .mimo_fit import fit_block_decoupled
from .sos_plant import DOFS, labelled_modes, build_plant, fit_suspension, SOSOptic

_I = {d: i for i, d in enumerate(DOFS)}


def oracle_by_dof() -> dict[str, tuple[float, float]]:
    """Per-DOF oracle ``{DOF: (f0, Q)}`` from the analytic plant's labelled eigenmodes.

    DOF-keyed rather than a bare frequency list: L (1.000 Hz) and T (0.998 Hz) are 1.8 mHz
    apart, so matching a recovered mode to the *nearest* oracle frequency silently swaps them.
    The recovery is scored against the mode of the DOF it actually came from.
    """
    optic = SOSOptic()
    return labelled_modes(build_plant(optic, fit_suspension(optic)), optic)


@dataclass
class ModeRecovery:
    """One recovered mode compared to the oracle."""
    dof: str                 # block label (e.g. "LP", "T")
    f0: float                # recovered resonance [Hz]
    Q: float                 # recovered quality factor
    f0_std: float            # CRB 1-sigma on f0 [Hz]
    Q_std: float             # CRB 1-sigma on Q
    f0_true: float           # nearest oracle mode [Hz]
    Q_true: float
    n_sigma_f0: float        # |f0 - f0_true| / f0_std
    n_sigma_Q: float

    @property
    def frac_err_f0(self) -> float:
        return abs(self.f0 - self.f0_true) / self.f0_true


@dataclass
class BandSpec:
    name: str
    fs: float
    nperseg: int
    n_periods: int
    drive: tuple[str, ...]              # DOF driven in this campaign
    blocks: tuple[tuple[tuple[str, ...], tuple[tuple[float, float], ...]], ...]
    anchors: tuple[float, ...]          # off-resonance baseline lines [Hz]
    n_transient: int = 1
    lines_per_mode: int = 25
    span_linewidths: float = 8.0
    noise_frac: float = 1e-3
    seed: int = 5


#: Prior mode frequencies used only to seed the fit (the campaign identifies, not asserts).
_SEED = {"L": 1.0, "P": 0.6, "T": 0.998, "Y": 0.7, "V": 16.0, "R": 22.63}

LOW_BAND = BandSpec(
    name="low", fs=32.0, nperseg=16384, n_periods=16,
    drive=("L", "P", "T", "Y"),
    blocks=((("L", "P"), ((1.0, 50.0), (0.6, 50.0))),
            (("T",), ((0.998, 50.0),)),
            (("Y",), ((0.7, 50.0),))),
    anchors=(0.4, 0.5, 0.85, 1.05, 1.15))

HIGH_BAND = BandSpec(
    name="high", fs=64.0, nperseg=8192, n_periods=16,
    drive=("V", "R"),
    blocks=((("V",), ((16.0, 50.0),)),
            (("R",), ((22.63, 50.0),))),
    anchors=(14.0, 15.0, 18.0, 20.0, 24.0, 26.0), seed=7)


def _excited_lines(spec: BandSpec, df: float) -> np.ndarray:
    lines = set()
    for d in spec.drive:
        f0 = _SEED[d]
        g = f0 / 50.0
        for k in np.linspace(-spec.span_linewidths, spec.span_linewidths, spec.lines_per_mode):
            lines.add(round((f0 + k * g) / df) * df)
    for f in spec.anchors:
        lines.add(round(f / df) * df)
    return np.array(sorted(x for x in lines if x > df))


def run_band(plant: ReducedStateSpacePlant, spec: BandSpec,
             oracle: dict[str, tuple[float, float]] | None = None) -> list[ModeRecovery]:
    """Run one focused campaign + block-decoupled recovery; return per-mode CRB comparisons."""
    oracle = oracle or oracle_by_dof()
    df = spec.fs / spec.nperseg
    freq_lines = _excited_lines(spec, df)

    fgrid = np.fft.rfftfreq(spec.nperseg, 1 / spec.fs)
    psd = np.zeros(len(fgrid))
    psd[[int(np.argmin(np.abs(fgrid - fl))) for fl in freq_lines]] = 1.0

    # absolute on-resonance noise floor (a broadband median|G| underflows off-resonance)
    onres = float(np.median([np.abs(plant.eval([_SEED[d]])).max() for d in spec.drive]))

    exc = {f"E{d}": f"m1.drive.{d}" for d in spec.drive}
    sns = {f"S{d}": f"m1.disp.{d}" for d in DOFS}
    be = ReducedPlantBackend(plant, exc, sns, fs=spec.fs,
                             sensor_asd=onres * spec.noise_frac, seed=spec.seed)
    exps, freq = assemble_campaign(
        be, [f"E{d}" for d in spec.drive], [f"E{d}" for d in spec.drive],
        [f"S{d}" for d in DOFS], freq_lines, fs=spec.fs, nperseg=spec.nperseg,
        n_periods=spec.n_periods, drive_psd=psd, n_transient=spec.n_transient, seed=spec.seed)

    aidx = {d: i for i, d in enumerate(spec.drive)}
    blocks = [{"sensors": [_I[d] for d in sd],
               "actuators": [aidx[d] for d in sd], "modes": list(modes)}
              for sd, modes in spec.blocks]
    block_dofs = [sd for sd, _ in spec.blocks]
    res = fit_block_decoupled(exps, freq, blocks, dof=spec.n_periods - spec.n_transient)

    out: list[ModeRecovery] = []
    for b, dofs in zip(res, block_dofs):
        label = "".join(dofs)
        # match each recovered mode to the oracle of the DOF it came from (within this block
        # only) — never to a nearer frequency in another block (the L/T doublet trap).
        avail = {d: oracle[d] for d in dofs}
        for i, (f0, Q) in enumerate(b["modes"]):
            d = min(avail, key=lambda dd: abs(avail[dd][0] - f0))
            f0_true, Qt = avail.pop(d)
            mu = b["mu"][i]
            sf, sq = mu["f0_std"], mu["Q_std"]
            out.append(ModeRecovery(
                dof=d, f0=f0, Q=Q, f0_std=sf, Q_std=sq,
                f0_true=f0_true, Q_true=Qt,
                n_sigma_f0=abs(f0 - f0_true) / sf if sf > 0 else np.inf,
                n_sigma_Q=abs(Q - Qt) / sq if sq > 0 else np.inf))
    return out


def run_full_recovery(plant: ReducedStateSpacePlant | None = None) -> list[ModeRecovery]:
    """Both bands: the full 6-DOF SOS recovery, one ModeRecovery per mode."""
    plant = plant or ReducedStateSpacePlant.load("sos", suffix="_6dof")
    oracle = oracle_by_dof()
    return run_band(plant, LOW_BAND, oracle) + run_band(plant, HIGH_BAND, oracle)


def _main() -> None:
    recs = run_full_recovery()
    print(f"{'DOF':4s} {'f0':>9s} {'f0_true':>9s} {'frac_err':>10s} {'nσ(f0)':>7s} "
          f"{'Q':>7s} {'nσ(Q)':>7s}")
    worst = 0.0
    for r in recs:
        worst = max(worst, r.n_sigma_f0, r.n_sigma_Q)
        print(f"{r.dof:4s} {r.f0:9.4f} {r.f0_true:9.4f} {r.frac_err_f0:10.2e} "
              f"{r.n_sigma_f0:7.2f} {r.Q:7.2f} {r.n_sigma_Q:7.2f}")
    print(f"\nworst n-sigma vs oracle: {worst:.2f}   "
          f"worst frac f0 error: {max(r.frac_err_f0 for r in recs):.2e}")


if __name__ == "__main__":
    _main()
