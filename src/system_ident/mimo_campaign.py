"""Assemble per-actuator sample-mean spectra + stacked covariances for the modal fit.

Robust method (P&S): n_exp = n_act -- drive each actuator separately with the periodic
multisine, read all drive monitors + sensors through the (closed) loop, and form, per excited
line: Ybar (sensor sample mean over periods), Ubar (drive-monitor sample mean), and Cz, the
sample covariance of the MEAN of the stacked [Y;U] vector. Mirrors
loop.SysIDLoop._estimate_tf_periodic but keeps ALL channels (no ratio). Verified end-to-end:
the rank-1 modal fit recovers a 6-DoF coupled plant's modal frequencies to ~0.15% through the
live damping loops.
"""
from __future__ import annotations
import numpy as np
from .excitation import multisine_from_psd


def _period_spectra(x, nperseg, n_transient):
    x = np.asarray(x, float)
    P = len(x) // nperseg
    X = np.fft.rfft(x[:P * nperseg].reshape(P, nperseg), axis=1)
    return X[n_transient:]                       # drop leading settling periods


def assemble_campaign(backend, exc_names, drive_names, sens_names, freq_lines, *,
                      fs, nperseg, n_periods, drive_psd, n_transient=1, seed=0):
    """Drive each actuator in turn; return (exps, freq).

    exps is a list of (Ybar, Ubar, Cz), one per excitation channel:
      Ybar (F, n_sens), Ubar (F, n_act) complex sample-mean spectra at the excited lines,
      Cz   (F, n_sens+n_act, n_sens+n_act) sample covariance of the stacked mean.
    """
    fs = float(fs)
    nperseg = int(nperseg)
    f = np.fft.rfftfreq(nperseg, 1.0 / fs)
    lines = np.array([int(np.argmin(np.abs(f - fl))) for fl in freq_lines])
    # DFT-resolution guard (A5): requested lines closer than df collide on the same rfft bin
    # and are silently merged/dropped — a resolution error the caller must see.
    if len(np.unique(lines)) != len(freq_lines):
        raise ValueError(f"{len(freq_lines)} requested lines collapse onto "
                         f"{len(np.unique(lines))} distinct rfft bins (df = {fs/nperseg:.5g} "
                         f"Hz) — space excited lines ≥ df apart or raise nperseg.")
    rng = np.random.default_rng(seed)
    duration = n_periods * nperseg / fs
    n_sens, n_act = len(sens_names), len(drive_names)
    exps = []
    for exc in exc_names:
        drive = multisine_from_psd(drive_psd, fs, nperseg, n_periods, f, seed=rng)
        backend.inject(exc, drive, fs)
        data = backend.read(list(drive_names) + list(sens_names), duration)
        Yp = np.stack([_period_spectra(data[s], nperseg, n_transient)[:, lines]
                       for s in sens_names], axis=-1)        # (P_eff, F, n_sens)
        Up = np.stack([_period_spectra(data[d], nperseg, n_transient)[:, lines]
                       for d in drive_names], axis=-1)        # (P_eff, F, n_act)
        Zp = np.concatenate([Yp, Up], axis=-1)               # (P_eff, F, n_sens+n_act)
        P_eff = Zp.shape[0]
        Zbar = Zp.mean(0)
        Cz = np.empty((len(lines), n_sens + n_act, n_sens + n_act), complex)
        for k in range(len(lines)):
            dk = Zp[:, k, :] - Zbar[k]
            Cz[k] = (dk.conj().T @ dk) / (P_eff - 1) / P_eff   # covariance of the mean
        exps.append((Zbar[:, :n_sens], Zbar[:, n_sens:], Cz))
        backend.ramp_down(exc, 1.0)
    return exps, np.asarray(freq_lines, float)
