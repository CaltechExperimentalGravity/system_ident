import numpy as np
import pytest
from system_ident.design.resolution import recommend_resolution


def test_recommend_resolution_from_ringdown():
    nperseg, df, ntr = recommend_resolution([(0.67, 50), (3.78, 50)], fs=256, bins_per_fwhm=4)
    fwhm_min = 0.67 / 50
    assert df <= fwhm_min / 4 * 1.01            # resolves the narrowest (lowest-f) peak
    assert nperseg & (nperseg - 1) == 0         # power of two
    assert np.isclose(df, 256 / nperseg)
    tau_max = 50 / (np.pi * 0.67)               # slowest ringdown
    assert ntr * (nperseg / 256) >= 3 * tau_max * 0.999


def test_recommend_resolution_validates():
    with pytest.raises(ValueError):
        recommend_resolution([], fs=256)
    with pytest.raises(ValueError):
        recommend_resolution([(1.0, -5)], fs=256)
