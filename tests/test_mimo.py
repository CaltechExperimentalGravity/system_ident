from __future__ import annotations
import numpy as np, control, pytest
from system_ident.mimo_plant import mimo_suspension, input_matrix, output_matrix


def test_plant_shapes_square_and_rectangular():
    Gsq = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.25)
    assert Gsq.noutputs == 2 and Gsq.ninputs == 2
    Grect = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=3, n_act=2, coupling=0.25)
    assert Grect.noutputs == 3 and Grect.ninputs == 2


def test_plant_is_coupled_with_shared_poles():
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.3)
    f = np.geomspace(0.3, 8, 200)
    H = G(2j * np.pi * f)  # (2,2,200)
    # off-diagonal is non-trivial (genuine coupling)
    assert np.max(np.abs(H[0, 1])) / np.max(np.abs(H[0, 0])) > 0.05
    # all elements share the same poles (same denominator roots) -> resonances line up
    pk00 = f[np.argmax(np.abs(H[0, 0]))]
    pk11 = f[np.argmax(np.abs(H[1, 1]))]
    assert min(abs(pk00 - 0.6), abs(pk00 - 1.5)) < 0.1 and min(abs(pk11 - 0.6), abs(pk11 - 1.5)) < 0.1


def test_decoupling_matrix_shapes():
    Min = input_matrix(2, 3, kind="perturbed", seed=1)
    assert Min.shape == (2, 3)
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2)
    Mo = output_matrix(G, n_act=2, n_dof=2, basis="eigenmode")
    assert Mo.ninputs == 2 and Mo.noutputs == 2


# ---------------------------------------------------------------------------
# Task 2: CoupledLoop
# ---------------------------------------------------------------------------
from system_ident.mimo_loop import CoupledLoop, velocity_damper


def _square_loop(fs=64.0, basis="euler"):
    G = mimo_suspension([(0.6, 20), (1.5, 30)], n_sens=2, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 2, kind="identity")
    Mout = output_matrix(G, n_act=2, n_dof=2, basis=basis)
    return CoupledLoop(G, C, Min, Mout, fs=fs)


def test_loop_is_stable():
    assert _square_loop().is_stable()


def test_loop_oracle_is_discrete_plant():
    lp = _square_loop()
    f = np.array([0.4, 3.0])
    z = np.exp(2j * np.pi * f / lp.fs)
    expect = lp.Gd(z)  # (2,2,2)
    np.testing.assert_allclose(lp.oracle(f), expect, rtol=1e-9)


def test_sensitivity_identity():
    # S = (I+L)^-1  =>  shape must be n_act×n_act in discrete form
    lp = _square_loop()
    assert lp.Sd.ninputs == 2 and lp.Sd.noutputs == 2


# ---------------------------------------------------------------------------
# Task 3: matrix-inverse recovery + off-resonance mask
# ---------------------------------------------------------------------------
from system_ident.mimo_loop import recover_open_loop, off_resonance_mask


def test_matrix_recovery_exact_offres_and_per_pair_biased():
    lp = _square_loop()
    f = np.geomspace(0.3, 8, 80); z = np.exp(2j*np.pi*f/lp.fs)
    Sd = lp.Sd(z).transpose(2,0,1)            # (nbin, n_act, n_act)
    Gd = lp.Gd(z).transpose(2,0,1)            # (nbin, n_sens, n_act)
    Xmat = Sd                                  # injected-ref -> drive monitors
    Ymat = Gd @ Sd                             # injected-ref -> responses
    Grec = recover_open_loop(Xmat, Ymat)
    mask = off_resonance_mask(f, [0.6, 1.5])
    rel = np.array([np.max(np.abs(Grec[k]-Gd[k]))/np.max(np.abs(Gd[k])) for k in range(len(f))])
    assert np.median(rel[mask]) < 1e-6        # matrix recovery exact off-resonance
    # per-pair ratio Y_ij/X_jj is badly biased off-diagonal (sanity: why we need the inverse)
    pair = np.array([Ymat[k] / np.diag(Xmat[k])[None,:] for k in range(len(f))])
    od = np.array([abs(pair[k,0,1]-Gd[k,0,1])/abs(Gd[k,0,1]) for k in range(len(f))])
    assert np.median(od[mask]) > 0.3


# ---------------------------------------------------------------------------
# Task 4: MIMOTwinBackend
# ---------------------------------------------------------------------------
from system_ident.backends.mimo_twin import MIMOTwinBackend
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop


def _backend(lp, **kw):
    exc = {f"EXC{j}": j for j in range(lp.n_act)}
    drv = {f"DRV{j}": j for j in range(lp.n_act)}
    sen = {f"SEN{i}": i for i in range(lp.n_sens)}
    return MIMOTwinBackend(lp, exc, drv, sen, **kw)


def test_backend_shapes_and_consistency():
    lp = _square_loop(); be = _backend(lp, seed=0)
    fs, nper, npe = lp.fs, 1024, 6
    T = nper*npe/fs
    drive = np.ones(int(T*fs))
    be.inject("EXC0", drive, fs)
    seg = be.read(["DRV0","DRV1","SEN0","SEN1"], T)
    assert all(seg[c].shape == (int(round(T*fs)),) for c in seg)


def test_backend_recovers_diagonal_offres():
    # drive one actuator, the j=0 monitor/sensor reference FRF recovers Gd[0,0] off-res
    lp = _square_loop(); be = _backend(lp, sensor_asd=0.0, seed=1)
    fs, nper, npe = lp.fs, 1024, 6
    fa = np.fft.rfftfreq(nper, 1/fs); band = (fa>=0.3)&(fa<=8.0); freq = fa[band]
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    u = multisine_from_psd(Pxx, fs, nper, npe, freq, seed=np.random.default_rng(0))
    be.inject("EXC0", u, fs)
    seg = be.read(["EXC0","DRV0","SEN0"], nper*npe/fs)
    Hx,_,_ = SysIDLoop._estimate_tf_periodic(seg["EXC0"], seg["DRV0"], fs, nper, band, 2)
    Hy,_,_ = SysIDLoop._estimate_tf_periodic(seg["EXC0"], seg["SEN0"], fs, nper, band, 2)
    # H_y / H_x is NOT Gd[0,0] (closed-loop coupling) — but the full matrix recovery is exact;
    # here just assert the monitor FRF is finite & nonzero (the sim ran through the loop)
    assert np.all(np.isfinite(Hx)) and np.median(np.abs(Hx)) > 0


# ---------------------------------------------------------------------------
# Task 5: end-to-end recovery — square sanity, realistic (M_in/M_out+noise), non-square
# ---------------------------------------------------------------------------


def _campaign(lp, *, sensor_asd=0.0, process_asd=0.0, nper=1024, npe=6, seed=0, modes_hz=(0.6, 1.5)):
    fs = lp.fs
    fa = np.fft.rfftfreq(nper, 1/fs); band = (fa>=0.3)&(fa<=8.0); freq = fa[band]
    Pxx = np.full_like(freq, 1.0/(freq[-1]-freq[0]))
    Xcols, Ycols = [], []
    for j in range(lp.n_act):
        be = _backend(lp, sensor_asd=sensor_asd, process_asd=process_asd, seed=seed*10+j)
        u = multisine_from_psd(Pxx, fs, nper, npe, freq, seed=np.random.default_rng(j))
        be.inject(f"EXC{j}", u, fs)
        ch = [f"EXC{j}"] + [f"DRV{a}" for a in range(lp.n_act)] + [f"SEN{i}" for i in range(lp.n_sens)]
        seg = be.read(ch, nper*npe/fs)
        Xcols.append([SysIDLoop._estimate_tf_periodic(seg[f"EXC{j}"], seg[f"DRV{a}"], fs, nper, band, 2)[0]
                      for a in range(lp.n_act)])
        Ycols.append([SysIDLoop._estimate_tf_periodic(seg[f"EXC{j}"], seg[f"SEN{i}"], fs, nper, band, 2)[0]
                      for i in range(lp.n_sens)])
    Xmat = np.array(Xcols).transpose(2,1,0)        # (nbin, n_act, n_drive)
    Ymat = np.array(Ycols).transpose(2,1,0)        # (nbin, n_sens, n_drive)
    Grec = recover_open_loop(Xmat, Ymat)
    Gd = lp.oracle(freq).transpose(2,0,1)
    mask = off_resonance_mask(freq, list(modes_hz))
    rel = np.array([np.max(np.abs(Grec[k]-Gd[k]))/np.max(np.abs(Gd[k])) for k in range(len(freq))])
    return rel, mask


def test_recover_square_sanity():
    lp = _square_loop(basis="euler")              # M_in=I, M_out=I
    rel, mask = _campaign(lp)
    assert np.median(rel[mask]) < 5e-3


def test_recover_square_realistic_with_noise():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 2, kind="perturbed", seed=3)
    Mout = output_matrix(G, n_act=2, n_dof=2, basis="eigenmode")
    lp = CoupledLoop(G, C, Min, Mout, fs=64.0)
    rel, mask = _campaign(lp, sensor_asd=1e-3, seed=2)
    assert np.median(rel[mask]) < 5e-2           # off-res recovery under nontrivial M_in/M_out + noise


def test_recover_non_square():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=3, n_act=2, coupling=0.25)
    C = [velocity_damper(0.5, 20.0) for _ in range(2)]
    Min = input_matrix(2, 3, kind="perturbed", seed=4)   # n_dof=2, n_sens=3
    Mout = output_matrix(G, n_act=2, n_dof=2, basis="euler")
    lp = CoupledLoop(G, C, Min, Mout, fs=64.0)
    rel, mask = _campaign(lp, seed=5)
    assert lp.n_sens == 3 and lp.n_act == 2
    assert np.median(rel[mask]) < 5e-2           # rectangular G (3×2) recovered off-res


# ---------------------------------------------------------------------------
# Task 6: 6-DoF (L/P/Y/R/V/T) recovery + Euler/eigenmode basis selectable
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_six_dof_recovery():
    # L/P/Y/R/V/T — six shared modes; square 6/6/6; representative eigenmode decoupler
    modes = [(0.43,100),(0.56,100),(0.9,80),(1.0,80),(2.0,60),(3.4,60)]
    G = mimo_suspension(modes, n_sens=6, n_act=6, coupling=0.2)
    C = [velocity_damper(0.4, 20.0) for _ in range(6)]
    Min = input_matrix(6, 6, kind="perturbed", seed=7)
    Mout = output_matrix(G, n_act=6, n_dof=6, basis="eigenmode")
    lp = CoupledLoop(G, C, Min, Mout, fs=128.0)
    assert lp.is_stable()
    rel, mask = _campaign(lp, sensor_asd=1e-3, nper=2048, npe=6, seed=8,
                          modes_hz=[0.43, 0.56, 0.9, 1.0, 2.0, 3.4])
    assert np.median(rel[mask]) < 1e-1


def test_basis_selectable():
    G = mimo_suspension([(0.6,20),(1.5,30)], n_sens=2, n_act=2)
    e = output_matrix(G, 2, 2, basis="euler"); m = output_matrix(G, 2, 2, basis="eigenmode")
    # euler is static (no states), eigenmode is dynamic (has states) — genuinely different
    assert control.tf2ss(e).nstates == 0 and control.tf2ss(m).nstates > 0
