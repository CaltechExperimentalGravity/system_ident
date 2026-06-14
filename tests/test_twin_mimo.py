"""MIMO coupling in the twin: a drive on one input must show up on every output
through the off-diagonal plant terms, measured leakage-free by the periodic FRF.
"""

import numpy as np
import scipy.signal as sig
import pytest

from system_ident.backends.twin import TwinBackend
from system_ident.excitation import multisine_from_psd
from system_ident.loop import SysIDLoop
from system_ident.model import TFModel
from system_ident.plant import SuspensionPlant

FS = 32.0
NPERSEG = 2048
N_PERIODS = 6
TOTAL = NPERSEG * N_PERIODS / FS

# A genuine 2x2 plant: two modes on the diagonal, cross terms off it.
H11 = TFModel.from_resonances([(0.8, 20.0)], 100.0)   # POS <- POS
H22 = TFModel.from_resonances([(1.4, 25.0)], 80.0)    # PIT <- PIT
H21 = TFModel.from_resonances([(0.8, 20.0)], 25.0)    # PIT <- POS  (cross)
H12 = TFModel.from_resonances([(1.4, 25.0)], 15.0)    # POS <- PIT  (cross)


def _grid():
    f_all = np.fft.rfftfreq(NPERSEG, 1 / FS)
    band = (f_all >= 0.1) & (f_all <= 5.0)
    return band, f_all[band]


def _disc_frf(tf, freq):
    b, a = sig.bilinear(tf.num, tf.den, FS)
    _, H = sig.freqz(b, a, worN=2 * np.pi * freq / FS)
    return H


def _twin():
    plant = SuspensionPlant({"POS": H11, "PIT": H22}, FS)
    coupling = {("PIT", "POS"): H21, ("POS", "PIT"): H12}
    return TwinBackend(
        plant, {"E1": "POS", "E2": "PIT"}, {"R1": "POS", "R2": "PIT"},
        fs=FS, sensor_asd=0.0, coupling=coupling,
    )


def _measure_column(tw, drive_ch, freq, band):
    """Drive one input, return the two measured output FRFs (this matrix column)."""
    Pxx = np.ones_like(freq)
    tw.inject(drive_ch, multisine_from_psd(Pxx, FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read([drive_ch, "R1", "R2"], TOTAL)
    H_to1, e1, _ = SysIDLoop._estimate_tf_periodic(seg[drive_ch], seg["R1"], FS, NPERSEG, band)
    H_to2, e2, _ = SysIDLoop._estimate_tf_periodic(seg[drive_ch], seg["R2"], FS, NPERSEG, band)
    return (H_to1, e1), (H_to2, e2)


def _rel_max(meas, err, tf, freq):
    exc = np.isfinite(err)
    truth = _disc_frf(tf, freq)[exc]
    return float((np.abs(meas[exc] - truth) / np.abs(truth)).max())


def test_twin_recovers_full_2x2_matrix():
    band, freq = _grid()

    # column 1: drive POS -> outputs (POS, PIT) recover (H11, H21)
    tw = _twin()
    (m11, e11), (m21, e21) = _measure_column(tw, "E1", freq, band)
    assert _rel_max(m11, e11, H11, freq) < 1e-2
    assert _rel_max(m21, e21, H21, freq) < 1e-2   # the cross term is real

    # column 2: drive PIT -> outputs (POS, PIT) recover (H12, H22)
    tw = _twin()
    (m12, e12), (m22, e22) = _measure_column(tw, "E2", freq, band)
    assert _rel_max(m12, e12, H12, freq) < 1e-2
    assert _rel_max(m22, e22, H22, freq) < 1e-2


def test_no_coupling_is_byte_identical_to_siso():
    band, freq = _grid()
    plant = SuspensionPlant({"POS": H11, "PIT": H22}, FS)
    tw = TwinBackend(plant, {"E1": "POS", "E2": "PIT"}, {"R1": "POS", "R2": "PIT"}, fs=FS)
    tw.inject("E1", multisine_from_psd(np.ones_like(freq), FS, NPERSEG, N_PERIODS, freq, seed=0), FS)
    seg = tw.read(["R2"], TOTAL)
    assert np.allclose(seg["R2"], 0.0)   # no drive on PIT, no coupling -> silent


def test_coupling_with_controllers_is_rejected():
    plant = SuspensionPlant({"POS": H11}, FS)
    with pytest.raises(NotImplementedError):
        TwinBackend(
            plant, {"E1": "POS"}, {"R1": "POS"}, fs=FS,
            controllers={"POS": ([1.0], [1.0])},
            coupling={("POS", "POS"): H11},
        )
