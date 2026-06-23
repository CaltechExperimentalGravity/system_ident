import numpy as np
from system_ident.mimo_campaign import assemble_campaign

class _FakeBackend:
    def __init__(self, fs, nperseg, n_periods, n_act, n_sens, sigma, seed=0):
        self.fs=fs; self.nperseg=nperseg; self.n_periods=n_periods
        self.n_act=n_act; self.n_sens=n_sens; self.sigma=sigma
        self.rng=np.random.default_rng(seed); self._d={}
    def inject(self, ch, ts, fs): self._d[ch]=np.asarray(ts,float)
    def ramp_down(self, ch, secs): pass
    def read(self, channels, duration):
        n=self.n_periods*self.nperseg; u=np.zeros((self.n_act,n))
        for ch,ts in self._d.items():
            j=int(ch[1:]); u[j,:min(len(ts),n)]=ts[:n]
        out={}
        for ch in channels:
            if ch.startswith("S"):
                i=int(ch[1:]); out[ch]=u[i % self.n_act]+self.rng.standard_normal(n)*self.sigma
            else:
                j=int(ch[1:]); out[ch]=u[j]+self.rng.standard_normal(n)*self.sigma
        return out

def test_assemble_campaign_shapes_and_psd():
    fs, nperseg, nper = 64.0, 256, 12
    f = np.fft.rfftfreq(nperseg, 1/fs); lines = np.array([4, 8, 12]); psd = np.zeros(len(f)); psd[lines]=1.0
    be = _FakeBackend(fs, nperseg, nper, 2, 2, 0.01, seed=3)
    exps, freq = assemble_campaign(be, ["E0","E1"], ["D0","D1"], ["S0","S1"], f[lines],
                                   fs=fs, nperseg=nperseg, n_periods=nper, drive_psd=psd, n_transient=1, seed=3)
    assert len(exps) == 2 and np.allclose(freq, f[lines])
    Yb, Ub, Cz = exps[0]
    assert Yb.shape == (3, 2) and Ub.shape == (3, 2) and Cz.shape == (3, 4, 4)
    assert np.allclose(Cz, np.conj(np.transpose(Cz, (0, 2, 1))), atol=1e-12)
    assert np.all(np.linalg.eigvalsh(Cz).real > -1e-9)
