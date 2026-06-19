"""Build the closed-loop ``x1hsts6dof`` composite the digital-twin's canonical
example drives, and expose it for the A3/A4 rungs.

The twin's ``twin/examples/sus_hsts_6dof`` closes the **real production L1-MC2
top-mass dampers** (``SUS-MC2_M1_DAMP_<dof>``) around the bare-M1 6×6 MIMO HSTS
plant, all six DOFs stable. This module reuses that example's ``lib.py`` verbatim
(plant + foton banks + per-DOF CAL) so the identification runs against *exactly*
the loop the twin ships — no re-derivation.

Everything here is **machine-specific glue** (the twin repo, its sibling
``aligo-suspension-models`` + ``aligo_filter_files`` archives, and a built
``x1hsts6dof`` are all local to the dev box). Callers guard on :func:`deps_available`
and skip when absent, exactly like the single-DOF real-model gate. ``system_ident``
itself never imports the twin — only this experiment helper does, lazily.

Plant note: ``x1hsts6dof``'s plant is **not** foton ZPK but a ``cdsStatespace``
block whose (A,B,C,D) are loaded at run time. The analytic oracle is therefore the
discretised state space (:func:`system_ident.backends.rtsfreerun_oracle.state_space_frf`),
not the scenario-YAML ZPK cascade the single-DOF ``x1hsts`` rungs use.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

# Machine-specific twin checkout (the one carrying the foton/SDF/.mat archives).
TWIN = Path("/Users/rana/GIT/digital_twin")
EXAMPLE = TWIN / "twin" / "examples" / "sus_hsts_6dof"


def deps_available() -> bool:
    """True iff the built model + twin example + production archives are all present."""
    return (
        importlib.util.find_spec("x1hsts6dof") is not None
        and (TWIN / "twin" / "src" / "twin").exists()
        and (EXAMPLE / "lib.py").exists()
        and (TWIN / "aligo_filter_files" / "l1" / "L1SUSMC2.txt").exists()
        and (TWIN / "aligo-suspension-models" / "hsts_full.mat").exists()
    )


def _import_lib():
    for p in (str(TWIN / "twin" / "src"), str(EXAMPLE)):
        if p not in sys.path:
            sys.path.insert(0, p)
    import lib  # twin/examples/sus_hsts_6dof/lib.py
    return lib


class HSTS6DOF:
    """A built ``x1hsts6dof`` with the real L1-MC2 dampers loaded; loops toggle-able.

    Channels (CDS ports verified against the built ``.mdl``): ``DRIVE_EXC_<dof>``
    (inject), ``READOUT_<dof>`` (sensor), ``MC2_M1_DAMP_<dof>_OUT`` (damper output
    = the controller feedback). There is no plant-input monitor port, so the true
    plant input is reconstructed as ``DRIVE_EXC − MC2_M1_DAMP_OUT``: the composite's
    ``COIL_DRV_SUM_<dof>`` junction is a ``"+-"`` sum (drive minus the delayed
    controller feedback), per ``digital_twin`` ``gen_x1hsts6dof.py``. The sign is
    negligible off-resonance (small feedback) but dominates at the damped resonances
    (large feedback) — getting it wrong biases exactly the modes being identified.
    """

    DAMP_CAL_SLOT = "FM1"  # free across every L1 MC2 bank; holds the per-DOF CAL scalar
    FEEDBACK_COEFF = -1.0  # COIL_DRV_SUM_<dof> is a "+-" (drive − feedback) junction

    def __init__(self):
        lib = _import_lib()
        from twin.foton_loader import apply_foton_bank
        from x1hsts6dof import x1hsts6dof

        self.lib = lib
        self.dofs = list(lib.DOFS)
        self.mdl = x1hsts6dof()
        self.fs_model = float(self.mdl.sample_rate)

        # Plant: bare-M1 6×6 HSTS state space → ZOH-discretise → load into the block.
        A, B, C, D = lib.load_plant_continuous()
        self.Ad, self.Bd, self.Cd, self.Dd = lib.discretise(A, B, C, D, self.fs_model)
        self.mdl.ss_set_abcd("HSTS_PLANT", self.Ad, self.Bd, self.Cd, self.Dd)

        # Dampers: real L1 MC2 foton banks, engaged FM list from the L1 SDF, with
        # the per-DOF CAL scalar in the free FM1 slot.
        self.fms = lib.load_site_swstat()
        for d in self.dofs:
            bank = self.bank(d)
            apply_foton_bank(self.mdl, lib.SITE_FOTON, f"{lib.BANK_PREFIX_SITE}_{d}",
                             bank, enable_sections=[f"FM{i}" for i in self.fms[d]])
            self.mdl.fm_set_zpk(bank, self.DAMP_CAL_SLOT, z=[], p=[], k=lib.SITE_BANK_CAL[d],
                                plane="s")

    # -- channel names -------------------------------------------------------
    def bank(self, d: str) -> str:       return f"{self.lib.BANK_PREFIX_MDL}_{d}"
    def exc(self, d: str) -> str:        return f"DRIVE_EXC_{d}"
    def readout(self, d: str) -> str:    return f"READOUT_{d}"
    def damp_out(self, d: str) -> str:   return f"{self.bank(d)}_OUT"
    def banks(self) -> list[str]:        return [self.bank(d) for d in self.dofs]

    # -- loop control --------------------------------------------------------
    def set_loops(self, on: bool) -> None:
        """Engage (``on``) or open every damper. Engaged = INPUT/OUTPUT + CAL + SDF FMs."""
        for d in self.dofs:
            bank = self.bank(d)
            if on:
                sw = {"INPUT": True, "OUTPUT": True, self.DAMP_CAL_SLOT: True}
                for i in self.fms[d]:
                    sw[f"FM{i}"] = True
                self.mdl.fm_set_switches(bank, **sw)
            else:
                self.mdl.fm_set_switches(bank, INPUT=False, OUTPUT=False)

    def reset(self) -> None:
        """Clear all damper history + reload the plant block (fair, repeatable start)."""
        self.mdl.fm_clear_history(*self.banks())
        self.mdl.ss_set_abcd("HSTS_PLANT", self.Ad, self.Bd, self.Cd, self.Dd)

    # -- oracle --------------------------------------------------------------
    def oracle_tensor(self, freq) -> np.ndarray:
        """Analytic open-loop FRF tensor ``H[k, out, in]`` from the discretised plant."""
        from system_ident.backends import rtsfreerun_oracle as orc
        return orc.state_space_frf(self.Ad, self.Bd, self.Cd, self.Dd, self.fs_model, freq)

    def oracle_prior(self, dof, *, fmin=0.3, fmax=8.0, max_modes=7, npts=500,
                     degen_tol=0.05, band_pad=1.5):
        """Parametric modal prior for ``dof``'s open-loop diagonal, fit from the
        analytic SS oracle, with the model **order chosen per DoF**.

        The six HSTS DoFs carry different numbers of in-band modes (L=5, T=3, V=2,
        R=4, P=5, Y=3). Fitting them all at a fixed high order *over-parameterises*
        the low-order ones — the extra pole pairs land out-of-band (V picks up 44 Hz /
        1.3 kHz ghosts) or collapse into a near-degenerate doublet (Y's 1.08/1.09),
        and those near-cancelling pole/zero pairs make the Fisher matrix singular →
        ``optimal_excitation`` → ``dispersion`` → ``pinv`` crashes with *SVD did not
        converge*. So we grow the order one mode at a time and stop **before** an order
        that (a) adds a mode outside ``[fmin/band_pad, fmax·band_pad]``, (b) adds a mode
        within ``degen_tol`` of another, or (c) no longer improves the oracle fit.
        """
        from system_ident.model import TFModel
        from system_ident.estimators.gml import GMLEstimator
        from system_ident.backends import rtsfreerun_oracle as orc

        j = self.dofs.index(dof)
        ff = np.geomspace(fmin, fmax, npts)
        G = self.oracle_tensor(ff)[:, j, j]
        err = np.full(len(ff), 1e-6) * np.abs(G)
        est = GMLEstimator()

        def fit(nm):
            fz = np.geomspace(0.4, 6.0, nm)
            poles = []
            for f in fz:
                w = 2 * np.pi * f
                poles += [-w / 40 + 1j * w, -w / 40 - 1j * w]
            den = np.real(np.poly(poles))
            num = np.zeros(len(den) - 2)
            num[-1] = abs(G[0]) * den[-1]
            return est.fit(ff, G, err, TFModel(num=num, den=den))

        def rel(model):
            return float(np.median(np.abs(model.eval(ff) - G) / np.abs(G)))

        def acceptable(model):
            ms = sorted(f for f, _ in orc.plant_modes(model))
            if any(f < fmin / band_pad or f > fmax * band_pad for f in ms):
                return False                                   # out-of-band ghost
            return all((b - a) / a >= degen_tol for a, b in zip(ms, ms[1:]))  # not degenerate

        best, best_rel = fit(1), None
        best_rel = rel(best)
        for nm in range(2, max_modes + 1):
            try:
                cand = fit(nm)
            except np.linalg.LinAlgError:
                break
            if not acceptable(cand):
                break
            r = rel(cand)
            if r >= best_rel * 0.95:                           # no meaningful gain → done
                break
            best, best_rel = cand, r
        return best

    # -- backend wiring ------------------------------------------------------
    def backend(self, drive_dof: str, *, fs, noise=None, warmup_s, seed, closed):
        """An :class:`RTSfreerunBackend` driving ``drive_dof`` on this prepared model.

        Sets the loops first, then exposes a virtual ``PLANT_IN_<dof>`` channel =
        ``DRIVE_EXC + MC2_M1_DAMP_OUT`` so a reference-based campaign recovers the
        open-loop plant through the closed loop. Reads back every DOF's sensor.
        """
        from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
        self.set_loops(closed)
        self.reset()
        exc = {self.exc(d): d for d in self.dofs}
        rb = {self.readout(d): d for d in self.dofs}
        plant_in = {self.plant_in(drive_dof): {"exc": self.exc(drive_dof),
                                               "feedback": [self.damp_out(drive_dof)],
                                               "feedback_coeff": self.FEEDBACK_COEFF}}
        return RTSfreerunBackend(mdl=self.mdl, exc_channels=exc, readback_channels=rb,
                                 plant_inputs=plant_in, noise=noise, fs=fs,
                                 warmup_s=warmup_s, seed=seed)

    def plant_in(self, d: str) -> str:   return f"PLANT_IN_{d}"

    # -- MIMO measurement ----------------------------------------------------
    def measure_tensor(self, *, closed, fs, nperseg, n_periods, band, freq,
                       px_total=1.0e7, n_passes=2, warmup_s=16.0, seed=0):
        """Reference-based FRF tensor ``H[out, in, k]`` over ``freq`` (band bins).

        Drives each input DOF with a P&S multisine (flat in-band budget — a
        broadband tensor sweep, not a single-mode optimal drive), reads every
        sensor and the driven DOF's reconstructed plant input, and forms the
        leakage-free reference FRF ``READOUT_i / PLANT_IN_j`` via the library's
        :meth:`SysIDLoop._estimate_tf_periodic`, inverse-variance-accumulated over
        ``n_passes``. With ``closed=True`` the plant input includes the damper
        feedback, so each **diagonal** element returns the *open-loop* plant
        (controller cancelled); off-diagonals under closed loop carry the
        cross-loop bias (the parametric joint MIMO fit — Track B — is the cure).
        """
        from system_ident.excitation import multisine_from_psd
        from system_ident.loop import SysIDLoop

        n = len(self.dofs)
        H = np.zeros((n, n, len(freq)), dtype=complex)
        Pxx = np.full(len(freq), px_total / (freq[-1] - freq[0]))
        total = nperseg * n_periods / fs
        for j, dj in enumerate(self.dofs):
            be = self.backend(dj, fs=fs, warmup_s=warmup_s, seed=seed, closed=closed)
            xch, routs = self.plant_in(dj), [self.readout(d) for d in self.dofs]
            accum = [dict(w=np.zeros(len(freq)), wH=np.zeros(len(freq), dtype=complex))
                     for _ in range(n)]
            for p in range(n_passes):
                drive = multisine_from_psd(Pxx, fs, nperseg, n_periods, freq,
                                           seed=np.random.default_rng(seed + 17 * p + j))
                be.inject(self.exc(dj), drive, fs)
                seg = be.read([xch, *routs], total)
                be.inject(self.exc(dj), np.zeros_like(drive), fs)
                for i, ri in enumerate(routs):
                    Hij, err, _ = SysIDLoop._estimate_tf_periodic(seg[xch], seg[ri],
                                                                  fs, nperseg, band)
                    H[i, j], _ = SysIDLoop._accumulate(accum[i], Hij, err)
        return H

    def measure_cancellation(self, dof, *, fs, nperseg, n_periods, band, freq,
                             px_total=1.0e7, n_passes=2, warmup_s=16.0, seed=0):
        """Closed-loop diagonal FRF two ways, to prove the controller is cancelled.

        Returns ``(H_ref, H_naive, G)``:
          * ``H_ref`` — reference-based ``READOUT/PLANT_IN`` (X = drive + feedback):
            recovers the **open-loop** plant ``G``.
          * ``H_naive`` — ``READOUT/DRIVE_EXC`` (X = injected drive only): the
            *suppressed* closed-loop response, biased by the sensitivity ``1/(1+L)``.
          * ``G`` — the analytic open-loop diagonal oracle for ``dof``.
        A real loop makes ``H_naive`` depart from ``G`` while ``H_ref`` tracks it.
        """
        from system_ident.excitation import multisine_from_psd
        from system_ident.loop import SysIDLoop

        j = self.dofs.index(dof)
        be = self.backend(dof, fs=fs, warmup_s=warmup_s, seed=seed, closed=True)
        xref, xexc, rout = self.plant_in(dof), self.exc(dof), self.readout(dof)
        Pxx = np.full(len(freq), px_total / (freq[-1] - freq[0]))
        total = nperseg * n_periods / fs
        acc_r = dict(w=np.zeros(len(freq)), wH=np.zeros(len(freq), dtype=complex))
        acc_n = dict(w=np.zeros(len(freq)), wH=np.zeros(len(freq), dtype=complex))
        Hr = Hn = None
        for p in range(n_passes):
            drive = multisine_from_psd(Pxx, fs, nperseg, n_periods, freq,
                                       seed=np.random.default_rng(seed + 31 * p))
            be.inject(xexc, drive, fs)
            seg = be.read([xref, xexc, rout], total)
            be.inject(xexc, np.zeros_like(drive), fs)
            hr, er, _ = SysIDLoop._estimate_tf_periodic(seg[xref], seg[rout], fs, nperseg, band)
            hn, en, _ = SysIDLoop._estimate_tf_periodic(seg[xexc], seg[rout], fs, nperseg, band)
            Hr, _ = SysIDLoop._accumulate(acc_r, hr, er)
            Hn, _ = SysIDLoop._accumulate(acc_n, hn, en)
        G = self.oracle_tensor(freq)[:, j, j]
        return Hr, Hn, G

    def parametric_recovery(self, dof, *, fs, nperseg, n_periods, band, freq,
                            px_total=1.0e7, n_passes=4, warmup_s=20.0, seed=0, closed=True):
        """A2-style optimal-excitation campaign recovering ``dof``'s diagonal plant.

        Seeds :meth:`oracle_prior` (auto per-DoF order), then runs the same
        ``run_siso_passes`` campaign the SISO examples use — prior-robust drive, then
        point-optimal refinement with CRB accumulation — against the reconstructed
        plant input (loops closed). Returns the per-pass history list.
        """
        import sys as _sys
        _docs = Path(__file__).resolve().parents[2] / "docs"
        if str(_docs) not in _sys.path:
            _sys.path.insert(0, str(_docs))
        from sysid_campaign import run_siso_passes

        prior = self.oracle_prior(dof)
        be = self.backend(dof, fs=fs, warmup_s=warmup_s, seed=seed, closed=closed)
        return run_siso_passes(be, self.exc(dof), self.readout(dof), prior,
                               x_ch=self.plant_in(dof), fs=fs, nperseg=nperseg,
                               n_periods=n_periods, band=band, freq=freq,
                               Pyy=np.ones_like(freq), px_total=px_total,
                               n_passes=n_passes, prior_uncertainty=0.3, seed=seed)

    def rel_err_tensor(self, H, freq):
        """Median per-element relative error ``|H − oracle| / |oracle|`` over ``freq``."""
        G = self.oracle_tensor(freq)                       # [k, out, in]
        n = len(self.dofs)
        M = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(n):
                g = G[:, i, j]
                M[i, j] = float(np.median(np.abs(H[i, j] - g) / np.abs(g)))
        return M
