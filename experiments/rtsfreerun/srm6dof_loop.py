"""SRM (Signal Recycling Mirror) 6-DOF closed-loop twin, for the joint MIMO fit.

The SRM is an HSTS suspension, so it shares the **same** bare-M1 6×6 HSTS plant
the L1-MC2 example (``hsts6dof_loop.HSTS6DOF``) drives — but it is closed by the
**real production L1-SRM** top-mass dampers (``SUS-SRM_M1_DAMP_<dof>``), not MC2's.

This mirrors :class:`hsts6dof_loop.HSTS6DOF` exactly, with three differences:

  1. Damper banks: read the SRM foton modules ``SRM_M1_DAMP_<dof>`` from
     ``L1SUSSRM.txt`` and write them into the compiled model's banks (which are
     baked in as ``MC2_M1_DAMP_<dof>`` — only the *label* is MC2; the contents are
     whatever we apply). Engaged FM list is the archived L1 SRM SDF state.
  2. CAL: MC2's per-DOF CAL scalars are tuned for MC2's bank shapes; SRM's banks
     differ, so SRM needs its own six CAL scalars (:data:`SRM_BANK_CAL`), tuned
     for tau ≈ 5 s ringdown per DOF on this plant.
  3. Backend: :meth:`backend` exposes a ``PLANT_IN_<d>`` monitor for **all six**
     DOFs (each = ``DRIVE_EXC_d − damper_d_OUT``), so a single campaign reads the
     full 6-wide X vector the rank-1 MIMO fit needs (HSTS6DOF only reconstructs the
     one driven DOF's input).

Everything else (plant load, discretise, loop toggle, oracle) is inherited from
``HSTS6DOF`` — only the SRM-specific foton source, FM list, and CAL are overridden.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import hsts6dof_loop as h6

TWIN = h6.TWIN
SRM_FOTON = TWIN / "aligo_filter_files" / "l1" / "L1SUSSRM.txt"
SRM_BANK_PREFIX_SITE = "SRM_M1_DAMP"   # foton-module prefix in L1SUSSRM.txt

# Per-DOF CAL knobs for the L1 SRM banks on the bare-M1 HSTS plant. Tuned
# 2026-06-22 for a per-DOF closed-loop ringdown time constant tau ≈ 5 s (the
# project "loosen up" target). Tuning method (same as the MC2 lib comment): build
# the full 6×6 SRM closure, impulse-drive each DOF, fit exp(-t/tau) to the 0.5-s
# peaks of the closed-loop response over t ∈ [3, 40] s, adjust CAL until stable
# and tau ≈ 5 s. Values are filled in by ``tune_srm_cal`` in the runner.
SRM_BANK_CAL: dict[str, float] = {
    "L": 1.0, "T": 1.0, "V": 1.0, "R": 1.0, "P": 1.0, "Y": 1.0,
}


def deps_available() -> bool:
    """True iff HSTS6DOF's deps are present *and* the SRM foton archive exists."""
    return h6.deps_available() and SRM_FOTON.exists()


def srm_engaged_fms() -> dict[str, list[int]]:
    """Engaged FM list per ``SUS-SRM_M1_DAMP_<dof>`` bank from the archived L1 SDF."""
    site = TWIN / "aligo_filter_files" / "l1"
    gps = (site / "archive_GPS").read_text().strip()
    sdf = json.loads((site / f"sdf_filter_state_{gps}.json").read_text())
    return {d: list(sdf["states"][f"SUS-{SRM_BANK_PREFIX_SITE}_{d}"]["fm_list"])
            for d in ("L", "T", "V", "R", "P", "Y")}


class SRM6DOF(h6.HSTS6DOF):
    """A built ``x1hsts6dof`` with the real **L1-SRM** dampers loaded; loops toggle-able.

    Same compiled composite, same shared HSTS plant, same ``MC2_M1_DAMP_<dof>``
    model-bank labels — but the bank *contents* are the SRM foton modules and the
    engaged FMs / CAL are SRM's. The backend exposes every ``PLANT_IN_<d>`` so a
    single drive sweep reads the full MIMO input vector.
    """

    def __init__(self, cal: dict[str, float] | None = None):
        lib = h6._import_lib()
        from twin.foton_loader import apply_foton_bank
        from x1hsts6dof import x1hsts6dof

        self.lib = lib
        self.dofs = list(lib.DOFS)
        self.mdl = x1hsts6dof()
        self.fs_model = float(self.mdl.sample_rate)
        self.cal = dict(SRM_BANK_CAL if cal is None else cal)

        # Plant: bare-M1 6×6 HSTS state space → ZOH-discretise → load into the block.
        # SAME plant as MC2 — the SRM *is* an HSTS.
        A, B, C, D = lib.load_plant_continuous()
        self.Ad, self.Bd, self.Cd, self.Dd = lib.discretise(A, B, C, D, self.fs_model)
        self.mdl.ss_set_abcd("HSTS_PLANT", self.Ad, self.Bd, self.Cd, self.Dd)

        # Dampers: real L1 SRM foton banks, SRM engaged FM list from the L1 SDF, with
        # the per-DOF CAL scalar in the free FM1 slot. The foton module is
        # SRM_M1_DAMP_<dof>; the model bank is MC2_M1_DAMP_<dof> (label is baked in).
        self.fms = srm_engaged_fms()
        for d in self.dofs:
            bank = self.bank(d)                      # MC2_M1_DAMP_<dof> (inherited)
            apply_foton_bank(self.mdl, SRM_FOTON, f"{SRM_BANK_PREFIX_SITE}_{d}",
                             bank, enable_sections=[f"FM{i}" for i in self.fms[d]])
            self.mdl.fm_set_zpk(bank, self.DAMP_CAL_SLOT, z=[], p=[], k=self.cal[d],
                                plane="s")

    def set_cal(self, cal: dict[str, float]) -> None:
        """Re-write the per-DOF CAL scalar into each bank's free FM1 slot in place.

        The compiled ``x1hsts6dof`` C++ model is a process singleton (it can only be
        instantiated once), so CAL tuning must mutate the existing model rather than
        rebuild it. Only the FM1 ZPK gain changes; the foton modules + engaged FMs
        loaded in ``__init__`` are untouched.
        """
        self.cal = dict(cal)
        for d in self.dofs:
            self.mdl.fm_set_zpk(self.bank(d), self.DAMP_CAL_SLOT, z=[], p=[],
                                k=self.cal[d], plane="s")

    # -- backend wiring: expose ALL six plant-input monitors ------------------
    def backend(self, drive_dof: str, *, fs, noise=None, warmup_s, seed, closed, ramp_s=3.0):
        """An :class:`RTSfreerunBackend` driving ``drive_dof``, exposing **every**
        ``PLANT_IN_<d>`` = ``DRIVE_EXC_d − damper_d_OUT`` (FEEDBACK_COEFF = −1).

        HSTS6DOF.backend only reconstructs the driven DOF's plant input; the MIMO
        fit needs the full 6-wide X vector, so we register all six virtual
        plant-input channels. Only ``drive_dof`` carries a non-zero injected drive
        per call; the other five PLANT_IN are pure feedback (the loop's response to
        the cross-coupled drive), which is exactly the off-diagonal X the joint fit
        uses.
        """
        from system_ident.backends.rtsfreerun_adapter import RTSfreerunBackend
        self.set_loops(closed)
        self.reset()
        exc = {self.exc(d): d for d in self.dofs}
        rb = {self.readout(d): d for d in self.dofs}
        plant_in = {
            self.plant_in(d): {"exc": self.exc(d),
                               "feedback": [self.damp_out(d)],
                               "feedback_coeff": self.FEEDBACK_COEFF}
            for d in self.dofs
        }
        return RTSfreerunBackend(mdl=self.mdl, exc_channels=exc, readback_channels=rb,
                                 plant_inputs=plant_in, noise=noise, fs=fs,
                                 warmup_s=warmup_s, seed=seed, ramp_s=ramp_s)

    # -- stability / CAL tuning helpers --------------------------------------
    def impulse_ringdown(self, dof: str, *, fs=256.0, dur=45.0, amp=1.0, warmup_s=2.0,
                         kick_s=0.2):
        """Drive ``dof`` with a short kick at the START, close all loops, return the
        ``dof`` readout time series (sysID rate) for stability / tau analysis.

        ``ramp_s=0`` so nothing is tiled into a trailing tail: the drive is the kick
        (length ``dur·fs``, zero after ``kick_s``) and the whole record is the free
        ringdown. The envelope should decay monotonically for a stable loop.
        """
        be = self.backend(dof, fs=fs, warmup_s=warmup_s, seed=0, closed=True, ramp_s=0.0)
        n = int(round(dur * fs))
        drive = np.zeros(n)
        drive[: max(1, int(round(kick_s * fs)))] = amp     # short kick at t=0
        be.inject(self.exc(dof), drive, fs)
        seg = be.read([self.readout(dof)], dur)
        return seg[self.readout(dof)]
