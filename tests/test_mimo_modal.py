"""Rank-1 modal MIMO fit: model/Jacobian, poles, peak-pick, end-to-end closed-loop recovery.

Comprehensive TDD coverage comes with the step-2 plan execution; these pin the verified core.
"""
import numpy as np
import pytest

from system_ident.mimo_modal import Rank1ModalModel
from system_ident.mimo_fit import (
    peak_pick_modes, init_residues, MIMOModalEstimator,
    parameter_covariance, modal_uncertainty, frf_band,
)


def test_jacobian_matches_finite_difference():
    m = Rank1ModalModel(3, 2, 2).set_reference(np.linspace(0.3, 3.0, 20))
    rng = np.random.default_rng(0)
    theta = rng.standard_normal(m.n_theta)
    # keep poles in the LHP-ish region (a>0, b>0)
    for k in range(m.n_modes):
        theta[k * m.per] = abs(theta[k * m.per]) + 0.5
        theta[k * m.per + 1] = abs(theta[k * m.per + 1]) + 0.1
    freq = np.linspace(0.3, 3.0, 20)
    J = m.jacobian(theta, freq)
    eps = 1e-7
    Jfd = np.zeros_like(J)
    for p in range(m.n_theta):
        dt = np.zeros(m.n_theta); dt[p] = eps
        Jfd[..., p] = (m.eval(theta + dt, freq) - m.eval(theta - dt, freq)) / (2 * eps)
    assert np.abs(J - Jfd).max() / np.abs(Jfd).max() < 1e-4


def test_eval_value_single_mode():
    m = Rank1ModalModel(2, 2, 1).set_reference(np.linspace(0.5, 2.0, 10))
    ab = m.ab_from_modes([(1.0, 10.0)])
    phi = np.array([[2.0, 0.5]]); psi = np.array([[1.0, 0.3]])
    theta = m.pack(ab, phi, psi)
    G = m.eval(theta, np.array([0.8]))
    sn = 2j * np.pi * 0.8 / m.s_ref
    D = sn * sn + ab[0][1] * sn + ab[0][0]
    assert np.isclose(G[0, 0, 0], 2.0 * 1.0 / D)
    assert np.isclose(G[0, 1, 1], 0.5 * 0.3 / D)


def test_pack_unpack_roundtrip():
    m = Rank1ModalModel(3, 2, 2)
    rng = np.random.default_rng(0)
    ab = [(rng.random()+0.5, rng.random()+0.1) for _ in range(2)]
    phi = rng.standard_normal((2, 3)); psi = rng.standard_normal((2, 2))
    theta = m.pack(ab, phi, psi)
    assert theta.shape == (m.n_theta,)
    got = m.unpack(theta)
    for k in range(2):
        assert np.isclose(got[k][0], ab[k][0]) and np.isclose(got[k][1], ab[k][1])
        assert np.allclose(got[k][2], phi[k]) and np.allclose(got[k][3], psi[k])


def test_set_reference_default_is_noop():
    m = Rank1ModalModel(2, 2, 1)
    assert m.s_ref == 1.0                      # default before set_reference


def test_rectangular_shapes():
    m = Rank1ModalModel(n_sens=3, n_act=2, n_modes=2).set_reference(np.linspace(0.3, 3.0, 20))
    assert m.n_theta == 2 * (2 + 3 + 2)
    theta = np.ones(m.n_theta)
    G = m.eval(theta, np.linspace(0.3, 3.0, 20))
    assert G.shape == (20, 3, 2)
    J = m.jacobian(theta, np.linspace(0.3, 3.0, 20))
    assert J.shape == (20, 3, 2, m.n_theta)


def test_poles_roundtrip_with_normalization():
    m = Rank1ModalModel(2, 2, 2).set_reference(np.linspace(0.2, 3.0, 50))
    ab = m.ab_from_modes([(0.6, 20.0), (1.5, 35.0)])
    phi = np.ones((2, 2)); psi = np.ones((2, 2))
    pq = m.poles(m.pack(ab, phi, psi))
    assert np.isclose(pq[0][0], 0.6, atol=1e-6) and np.isclose(pq[0][1], 20.0, rtol=1e-4)
    assert np.isclose(pq[1][0], 1.5, atol=1e-6) and np.isclose(pq[1][1], 35.0, rtol=1e-4)


def test_peak_pick_finds_modes():
    freq = np.linspace(0.3, 3.0, 200)
    m = Rank1ModalModel(2, 2, 2).set_reference(freq)
    ab = m.ab_from_modes([(0.7, 40.0), (1.8, 50.0)])
    phi = np.array([[1.0, 0.3], [0.2, 1.0]]); psi = np.array([[1.0, 0.1], [0.1, 1.0]])
    G = m.eval(m.pack(ab, phi, psi), freq)
    found = sorted(f for f, _ in peak_pick_modes(G, freq, 2))
    assert abs(found[0] - 0.7) < 0.05 and abs(found[1] - 1.8) < 0.05


@pytest.mark.slow
def test_closed_loop_modal_recovery_6dof():
    from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix
    from system_ident.mimo_loop import CoupledLoop, velocity_damper, recover_open_loop
    from system_ident.backends.mimo_twin import MIMOTwinBackend
    from system_ident.mimo_campaign import assemble_campaign

    fs, nperseg, nper = 128.0, 4096, 12
    modes = [(0.45, 20), (0.6, 25), (0.8, 18), (1.0, 30), (1.5, 35), (2.2, 28)]
    truth = sorted(mm[0] for mm in modes)
    plant = mimo_suspension(modes, n_sens=6, n_act=6, coupling=0.15, gain=100.0, seed=0)
    loop = CoupledLoop(plant, [velocity_damper(1.0, 4.0) for _ in range(6)],
                       input_matrix(6, 6), output_matrix(plant, 6, 6, basis="euler"), fs=fs)
    be = MIMOTwinBackend(loop, {f"E{j}": j for j in range(6)}, {f"D{j}": j for j in range(6)},
                         {f"S{i}": i for i in range(6)}, sensor_asd=1e-3, process_asd=1e-4, seed=7)
    f = np.fft.rfftfreq(nperseg, 1 / fs); lines = np.flatnonzero((f >= 0.3) & (f <= 2.6))
    psd = np.zeros(len(f)); psd[lines] = 1.0
    exps, freq = assemble_campaign(
        be, [f"E{j}" for j in range(6)], [f"D{j}" for j in range(6)],
        [f"S{i}" for i in range(6)], f[lines], fs=fs, nperseg=nperseg, n_periods=nper,
        drive_psd=psd, n_transient=3, seed=7)
    Xmat = np.stack([exps[l][1] for l in range(6)], -1)
    Ymat = np.stack([exps[l][0] for l in range(6)], -1)
    Gnp = recover_open_loop(Xmat, Ymat)
    m = Rank1ModalModel(6, 6, 6).set_reference(freq)
    ab = m.ab_from_modes(peak_pick_modes(Gnp, freq, 6))
    phi, psi = init_residues(m, ab, exps, freq)
    res = MIMOModalEstimator(m).fit(exps, freq, m.pack(ab, phi, psi))
    fit = sorted(p[0] for p in m.poles(res.theta))
    assert len(fit) == 6
    for ff, tt in zip(fit, truth):
        assert abs(ff - tt) / tt < 0.01          # all 6 modal f0 within 1% through the loop
    Ct = parameter_covariance(res, dof=nper - 3, n_sens=6)
    band = frf_band(m, res.theta, Ct, freq)
    assert band.shape == (len(freq), 6, 6) and np.all(np.isfinite(band))
    mu = modal_uncertainty(m, res.theta, Ct)
    assert all(d["f0_std"] > 0 for d in mu)
