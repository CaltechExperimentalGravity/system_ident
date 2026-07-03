import numpy as np
import pytest
from system_ident.mimo_modal import Rank1ModalModel
from system_ident.mimo_fit import (peak_pick_modes, find_modes, init_residues, initial_theta,
                                    MIMOModalEstimator, parameter_covariance,
                                    mimo_parameter_covariance, mimo_fisher_matrix,
                                    modal_uncertainty, modal_frac_uncertainty, frf_band,
                                    validate_fit, fit_block_decoupled)

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
        assert np.allclose(R, Rh, atol=1e-3)

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

def test_validate_fit_metrics():
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    modes = [(0.6,30),(1.6,40)]
    exps, freq, theta_true, G = synth_openloop(m, modes, phi, psi, sigZ=5e-3, M=20, seed=3)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    rep = validate_fit(m, res.theta, exps, freq, dof=20, modes_hz=[mm[0] for mm in modes])
    assert rep["frf_rel_median_offres"] < 0.1
    assert 0.3 < rep["cost_ratio"] < 3.0

def test_nonsquare_recovery():
    m = Rank1ModalModel(n_sens=4, n_act=2, n_modes=2)   # rectangular
    phi = np.array([[1.,.3,.2,.1],[.2,1.,.4,.3]]); psi = np.array([[1.,.2],[.1,1.]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=4)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 1e-2 and abs(fit[1]-1.6) < 1e-2

def test_prior_independent_recovery():
    # peak-pick is data-driven: recovery does not depend on the prior at all
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1.,.3,.2],[.2,1.,.4]]); psi = np.array([[1.,.2,.1],[.1,1.,.3]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6,30),(1.6,40)], phi, psi, sigZ=5e-3, M=20, seed=5)
    # wildly wrong prior (50% off) is ignored because peak-pick reads modes from G
    th0 = initial_theta(m, exps, freq, G, prior_modes=[(0.3, 20), (2.4, 20)])
    res = MIMOModalEstimator(m).fit(exps, freq, th0)
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert abs(fit[0]-0.6) < 1e-2 and abs(fit[1]-1.6) < 1e-2


def test_parameter_covariance_rejects_insufficient_dof():
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1., .3, .2], [.2, 1., .4]]); psi = np.array([[1., .2, .1], [.1, 1., .3]])
    exps, freq, _, G = synth_openloop(m, [(0.6, 30), (1.6, 40)], phi, psi, sigZ=5e-3, M=20, seed=6)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    with pytest.raises(ValueError):
        parameter_covariance(res, dof=m.n_sens + 1, n_sens=m.n_sens)   # d=1 < 2


def test_pole_prior_anchors_to_design_frequency():
    # A mode seeded at the WRONG frequency is pulled to its design prior (anchoring on).
    m = Rank1ModalModel(2, 2, 2)
    phi = np.array([[1., .3], [.2, 1.]]); psi = np.array([[1., .2], [.1, 1.]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6, 40), (1.6, 40)], phi, psi,
                                               sigZ=2e-3, M=20, seed=2)
    th0 = initial_theta(m, exps, freq, G, prior_modes=[(0.6, 40), (1.20, 40)])  # mode 1 seeded wrong
    res = MIMOModalEstimator(m).fit(exps, freq, th0,
                                    pole_prior_hz=[0.6, 1.6], prior_weight=1e2)
    f0 = sorted(p[0] for p in m.poles(res.theta))
    assert abs(f0[1] - 1.6) < 0.05            # anchored + data agree -> reaches 1.6, not the 1.2 seed

    import pytest
    with pytest.raises(ValueError):
        MIMOModalEstimator(m).fit(exps, freq, th0, pole_prior_hz=[0.6], prior_weight=1.0)  # wrong count


def test_mimo_fisher_independent_matches_post_fit():
    # The fit-independent MIMO CRB (from model+theta+exps) must reproduce the post-fit CRB
    # (from FitResult.jac) exactly, and yield a sensible scalar DONE criterion.
    m = Rank1ModalModel(3, 3, 2)
    phi = np.array([[1., .3, .2], [.2, 1., .4]]); psi = np.array([[1., .2, .1], [.1, 1., .3]])
    exps, freq, theta_true, G = synth_openloop(m, [(0.6, 30), (1.6, 40)], phi, psi,
                                               sigZ=3e-3, M=20, seed=5)
    res = MIMOModalEstimator(m).fit(exps, freq, initial_theta(m, exps, freq, G))
    Cfit = parameter_covariance(res, dof=18, n_sens=3)
    Cind = mimo_parameter_covariance(m, res.theta, exps, freq, dof=18, n_sens=3)
    np.testing.assert_allclose(Cfit, Cind, rtol=1e-9, atol=0)      # fit-independent == post-fit
    fisher = mimo_fisher_matrix(m, res.theta, exps, freq)
    assert fisher.shape == (m.n_theta, m.n_theta)
    fu = modal_frac_uncertainty(m, res.theta, Cind)
    assert 0.0 < fu < 1.0                                          # sensible DONE scalar
    # CRB can be evaluated at ANY theta with no fit (the ideal-bound-at-truth use case)
    assert modal_frac_uncertainty(m, theta_true,
                                  mimo_parameter_covariance(m, theta_true, exps, freq,
                                                            dof=18, n_sens=3)) > 0.0


def test_find_modes_data_driven_no_count_no_clustering():
    # A 4-DOF FRF on a FINE grid with 3 well-separated modes, each dominant in a DIFFERENT
    # channel and ONE much weaker in the summed power. find_modes must (a) return exactly 3
    # (order from the data, not supplied), (b) NOT cluster several onto one peak's bins
    # (the old peak_pick failure), and (c) find the weak-in-sum mode via its own channel.
    m = Rank1ModalModel(4, 4, 3)
    phi = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 0, 0.12]])   # mode 2 small, chan 3
    psi = np.array([[1., 0, 0, 0], [0, 1., 0, 0], [0, 0, 0, 0.12]])
    freq = np.linspace(0.3, 3.0, 1200)                               # fine grid
    _, _, theta, G = synth_openloop(m, [(0.6, 50), (1.3, 50), (2.4, 50)], phi, psi, freq=freq)
    found = find_modes(G, freq)
    f0 = sorted(f for f, _ in found)
    assert len(found) == 3, f"expected 3 modes, got {len(found)}: {f0}"
    for tf in (0.6, 1.3, 2.4):
        assert min(abs(f - tf) for f in f0) < 0.02, f"missed {tf}: {f0}"
    # the old strongest-bins pick clusters on a fine grid; find_modes must not
    assert all(abs(f0[i + 1] - f0[i]) > 0.1 for i in range(len(f0) - 1))


def test_fit_block_decoupled_resolves_spatial_doublet():
    # A 4-DOF plant that decouples into blocks {0,1} and {2,3}, with a near-coincident
    # frequency pair (1.000 Hz in block A, 1.004 Hz in block B; Q=50 -> 2% FWHM overlaps
    # the 0.4% split). The two modes are spatially orthogonal (each residue lives in one
    # block), so block-decoupled fitting resolves them with no frequency super-resolution.
    m = Rank1ModalModel(4, 4, 2)
    phi = np.array([[1.0, 0.5, 0.0, 0.0],      # mode 0 -> sensors {0,1}
                    [0.0, 0.0, 1.0, 0.6]])     # mode 1 -> sensors {2,3}
    psi = np.array([[1.0, 0.4, 0.0, 0.0],      # mode 0 -> actuators {0,1}
                    [0.0, 0.0, 1.0, 0.5]])     # mode 1 -> actuators {2,3}
    freq = np.linspace(0.85, 1.15, 220)
    exps, freq, theta, G = synth_openloop(m, [(1.000, 50), (1.004, 50)], phi, psi,
                                          sigZ=1e-3, M=20, seed=3, freq=freq)
    blocks = [{"sensors": [0, 1], "actuators": [0, 1], "modes": [(1.000, 50)]},
              {"sensors": [2, 3], "actuators": [2, 3], "modes": [(1.004, 50)]}]
    res = fit_block_decoupled(exps, freq, blocks, dof=18)
    fA = sorted(f for f, _ in res[0]["modes"])
    fB = sorted(f for f, _ in res[1]["modes"])
    assert abs(fA[0] - 1.000) < 1e-3            # block A recovers the 1.000 Hz member
    assert abs(fB[0] - 1.004) < 1e-3            # block B recovers the 1.004 Hz member
    assert res[0]["mu"] is not None and res[0]["mu"][0]["f0_std"] < 1e-3   # CRB populated
