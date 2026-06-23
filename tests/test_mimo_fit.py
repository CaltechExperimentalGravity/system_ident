import numpy as np
import pytest
from system_ident.mimo_modal import Rank1ModalModel
from system_ident.mimo_fit import (peak_pick_modes, init_residues, initial_theta,
                                    MIMOModalEstimator, parameter_covariance,
                                    modal_uncertainty, frf_band)

def synth_openloop(model, modes, phi, psi, *, sigZ=0.0, M=20, seed=0, freq=None):
    """Robust-method campaign on G = model truth (open loop). Returns (exps, freq, theta_true, G)."""
    if freq is None:
        freq = np.linspace(0.3, 3.0, 120)
    model.set_reference(freq)
    ab = model.ab_from_modes(modes)
    theta = model.pack(ab, phi, psi)
    G = model.eval(theta, freq); s = 2j * np.pi * freq
    rng = np.random.default_rng(seed); nsa = model.n_sens + model.n_act; exps = []
    for l in range(model.n_act):
        U = np.zeros((len(freq), model.n_act), complex); U[:, l] = 1.0 + 0.3 * np.cos(s.imag)
        Y = np.einsum('fij,fj->fi', G, U); per = []
        for _ in range(max(M, 2)):
            nY = (rng.standard_normal((len(freq), model.n_sens)) + 1j*rng.standard_normal((len(freq), model.n_sens))) * sigZ/np.sqrt(2)
            nU = (rng.standard_normal((len(freq), model.n_act)) + 1j*rng.standard_normal((len(freq), model.n_act))) * sigZ/np.sqrt(2)
            per.append(np.concatenate([Y + nY, U + nU], axis=1))
        per = np.array(per); Zb = per.mean(0)
        Cz = np.empty((len(freq), nsa, nsa), complex)
        for k in range(len(freq)):
            d = per[:, k, :] - Zb[k]; Cz[k] = (d.conj().T @ d) / (len(per) - 1) / len(per)
        exps.append((Zb[:, :model.n_sens], Zb[:, model.n_sens:], Cz))
    return exps, freq, theta, G

def test_peak_pick_finds_two_modes():
    m = Rank1ModalModel(2, 2, 2)
    freq = np.linspace(0.3, 3.0, 200); m.set_reference(freq)
    phi = np.array([[1.0, 0.2], [0.3, 1.0]]); psi = np.array([[1.0, 0.1], [0.2, 1.0]])
    _, _, theta, G = synth_openloop(m, [(0.7, 40), (1.8, 50)], phi, psi, freq=freq)
    found = sorted(f for f, _ in peak_pick_modes(G, freq, 2))
    assert abs(found[0] - 0.7) < 0.05 and abs(found[1] - 1.8) < 0.05

def test_init_residues_recovers_rank1_shapes():
    m = Rank1ModalModel(3, 2, 2)
    phi = np.array([[1.0, 0.4, 0.2], [0.1, 1.0, 0.5]]); psi = np.array([[1.0, 0.3], [0.2, 1.0]])
    exps, freq, theta, G = synth_openloop(m, [(0.6, 30), (1.6, 40)], phi, psi, sigZ=1e-4, M=20, seed=1)
    ab = m.ab_from_modes([(0.6, 30), (1.6, 40)])
    phi_hat, psi_hat = init_residues(m, ab, exps, freq)
    # residue matrices match up to per-mode sign/scale gauge -> compare R_k = phi psi^T
    for k in range(2):
        R = np.outer(phi[k], psi[k]); Rh = np.outer(phi_hat[k], psi_hat[k])
        assert np.allclose(R, Rh, atol=1e-2)

def test_estimator_recovers_poles_from_perturbed_init():
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.0,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=2)
    th0 = initial_theta(m, exps, freq, G)              # data-driven (peak-pick)
    res = MIMOModalEstimator(m).fit(exps, freq, th0)
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 5e-3 and abs(fit[1]-1.6) < 5e-3

@pytest.mark.slow
def test_crb_matches_monte_carlo():
    m = Rank1ModalModel(2, 2, 2)
    phi = np.array([[1.,.3],[.2,1.]]); psi = np.array([[1.,.2],[.1,1.]])
    modes = [(0.6,30),(1.6,40)]; M = 24
    e0, freq, th_t, G = synth_openloop(m, modes, phi, psi, sigZ=0.01, M=M, seed=1)
    r0 = MIMOModalEstimator(m).fit(e0, freq, initial_theta(m, e0, freq, G))
    Ct = parameter_covariance(r0, dof=M, n_sens=2)
    pred = modal_uncertainty(m, r0.theta, Ct)[0]["f0_std"]
    f0s = []
    for sd in range(40):
        e2, _, _, G2 = synth_openloop(m, modes, phi, psi, sigZ=0.01, M=M, seed=100+sd)
        r2 = MIMOModalEstimator(m).fit(e2, freq, initial_theta(m, e2, freq, G2))
        pq = m.poles(r2.theta)
        if len(pq) == 2 and abs(pq[0][0]-0.6) < 0.1: f0s.append(pq[0][0])
    mc = np.std(f0s)
    assert 0.4 < pred/mc < 2.5                         # CRB brackets the Monte-Carlo spread
    band = frf_band(m, r0.theta, Ct, freq)
    assert band.shape == (len(freq), 2, 2) and np.all(np.isfinite(band)) and np.all(band >= 0)
