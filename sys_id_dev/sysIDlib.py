"""Frequency-domain system-identification engine (SISO Pintelon–Schoukens reference).

The original optimal-excitation / maximum-likelihood-fit routines, first written
Jan 2024. This is the reference implementation the packaged pipeline is validated
against and ported from (see ``system_ident.estimators.invfreqs``).

Authors: Hang Yu (hang.yu2@montana.edu), Nathan Holland, Rana X. Adhikari.

Copyright (C) 2024 Hang Yu, Nathan Holland, Rana X. Adhikari.
Released under the GNU General Public License v3.0-or-later; see the LICENSE file
at the repository root.
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ
import scipy.signal as sig
from scipy.ndimage import gaussian_filter1d
import h5py as h5
import os.path
import matplotlib.ticker as ticker


########################################
###         model setup              ###
########################################

def unpack_par_dict(par_dict):
    """
    unpack the par_dict into a numpy array that can be easily looped over
    """
    par = np.hstack([
        par_dict['num'],
        par_dict['den']
    ])
    n_num = len(par_dict['num'])
    n_den = len(par_dict['den'])
    return par, n_num, n_den

def pack_par_to_dict(par, n_num, n_den):
    """
    inverting unpack_par_dict
    """
    
    num = par[:n_num]
    den = par[n_num:]
    
    par_dict = {'num':num,
                'den':den
               }
    return par_dict

def par_dict_to_TF_vect(freq, par_dict):
    """
    compute the TF given by par_dict
    at a given frequncy grid freq
    """
    par, n_num, n_den = unpack_par_dict(par_dict)
    num = par[:n_num]
    den = par[n_num:]
    
    __, GG = sig.freqs(num, den, worN=2.*np.pi*freq)
    return GG

def par_dict_to_sos(freq, par_dict, fs):
    """
    get digital filters in sos format based on par_dict
    """
    par, n_num, n_den = unpack_par_dict(par_dict)
    num = par[:n_num]
    den = par[n_num:]
    
    num_z, den_z = sig.bilinear(num, den, fs)
    sos = sig.tf2sos(num_z, den_z)
    return sos

def par_to_TF_vect(freq, par, n_num, n_den):
    """
    compute the TF given by par and n_num, n_den
    at a given frequncy grid freq
    """
    num = par[:n_num]
    den = par[n_num:]
    
    _, GG = sig.freqs(num, den, worN = 2*np.pi*freq)
    return GG
    
def par_to_sos(freq, par, n_num, n_den, fs):
    """
    get digital filters in sos format based on par_dict
    """
    num = par[:n_num]
    den = par[n_num:]
    
    num_z, den_z = sig.bilinear(num, den, fs)
    sos = sig.tf2sos(num_z, den_z)
    return sos

########################################
###         fisher info              ###
########################################    
    
def get_Fisher_from_psd(freq, par_dict,
                        Pxx, Pyy,
                        T_tot=None, n_avg=1,
                        dpar_dict=None, logflag_dict=None,
                        return_gamma_vs_freq=False):
    """
    Suppose we do a TF measurement of model G: 
            -----
        x ->| G |-> y
            -----        
    We compute here the Fisher matrix of G's parameters.  

    We use expression given in 
    https://dcc.ligo.org/LIGO-G2101503, p19.

    This assumes we already have some prior knowledge of G, 
    and our estimation of G can be parameterized by par_dict
    [whose format should follow unpack_par_dict()].

    Pxx is the PSD of the excitation x
    Pyy is the PSD of the readout when there is NO excitation (ie, quiet-time PSD of the readout)
    Both are evaluated at the freq grid 

    T_tot is the total time in [s] of the measurement.

    Alternatively, one can specify the number of averages n_avg = T_tot/T_perseg (no overlap)
    In this case, freq is assume uniformed spaced and df = 1/T_perseg
    Note, n_avg will be bypassed if T_tot is given.

    One can choose to manually set stepsize for each par when computing the Fisher numerically.
    This is done via the dpar_dict.
    It should be filled following the same order as par_dict.
    If dpar_dict=None, default stepsize of 1e-8 will be used on all the par.

    One can also choose to compute the absolute error or the fractional one using
        logflag_dict 
    (following the structure of par_dict; a par with flag>0 will return fractional error).
    If logflag_dict=None, it will return fractional error for ks=par[-1] and absolute error for the rest.
    """

    par, n_num, n_den = unpack_par_dict(par_dict)
    n_par = len(par)
    n_bin = len(freq)
    
    par /= par[n_num] # this makes sure the first element in den is fixed to 1

    if T_tot is None:
        df = freq[1]-freq[0]
        T_tot = (1./df) * n_avg

    if dpar_dict is None:
        dpar = np.ones(n_par) * 1.e-8
    else:
        dpar, _, _ = unpack_par_dict(dpar_dict)

    if logflag_dict is None:
        logflag = -np.ones(n_par, dtype=np.int)
        logflag[-1] = 1
    else:
        logflag, _, _ = unpack_par_dict(dpar_dict)

    dG_dpar = np.zeros([n_par, n_bin], dtype=np.complex128)
    for i in range(n_par):
        
        # skip the first element in den
        if i==n_num:
            continue
            
        par_u = par.copy()
        par_u[i] += dpar[i]
        GG_u = par_to_TF_vect(freq, par_u, n_num, n_den)

        par_l = par.copy()
        par_l[i] -= dpar[i]
        GG_l = par_to_TF_vect(freq, par_l, n_num, n_den)

        dG_dpar[i, :] = (GG_u-GG_l)/(2.*dpar[i])

        if logflag[i] > 0:
            dG_dpar[i, :] *= par[i]

    gamma = np.zeros([n_par, n_par])
    for i in range(n_par):
        for j in range(i, n_par, 1):
            gamma[i, j] = 2.*integ.trapezoid(
                np.real(np.conj(dG_dpar[i, :])*dG_dpar[j, :])*Pxx/Pyy,
                freq)

    for i in range(n_par):
        for j in range(i):
            gamma[i, j] = gamma[j, i]

    # delete the row/column corresponding to the first element in den
    gamma = np.delete(gamma, (n_num), axis=0)
    gamma = np.delete(gamma, (n_num), axis=1)
    
    gamma *= T_tot

    if return_gamma_vs_freq:
        gamma_vs_freq = np.zeros([n_par, n_par, n_bin])

        for i in range(n_par):
            for j in range(i, n_par, 1):
                gamma_vs_freq[i, j, :] =\
                    2.*np.real(np.conj(dG_dpar[i, :])*dG_dpar[j, :])*Pxx/Pyy

        for i in range(n_par):
            for j in range(i):
                gamma_vs_freq[i, j, :] = gamma_vs_freq[j, i, :]
        
        # delete the row/column corresponding to the first element in den
        gamma_vs_freq = np.delete(gamma_vs_freq, (n_num), axis=0)
        gamma_vs_freq = np.delete(gamma_vs_freq, (n_num), axis=1)

        gamma_vs_freq *= T_tot
        return gamma, gamma_vs_freq

    else:
        return gamma
    
    
def get_dispersion(freq, par_dict,
                   Pxx, Pyy,
                   T_tot=None, n_avg=1,
                   dpar_dict=None, logflag_dict=None,
                   return_gamma=False):
    """
    Compute the dispersion function following Pintelon & Schoukens
    Sec. 5.4.2

    Note one needs to specify T_tot only if one cares about the proper value of gamma.
    If one just want the optimized PSD, T_tot can be simply set to 1. 
    """
    n_bin = len(freq)
    gamma, gamma_vs_freq = \
        get_Fisher_from_psd(freq, par_dict,
                            Pxx, Pyy,
                            T_tot=None, n_avg=1,
                            dpar_dict=dpar_dict, logflag_dict=logflag_dict,
                            return_gamma_vs_freq=True)
    sigma = np.linalg.inv(gamma)
    nu = np.zeros(n_bin)

    Pxx_tot = np.sum(Pxx)

    for i in range(n_bin):
        gamma_loc = gamma_vs_freq[:, :, i] * Pxx_tot / Pxx[i]
        nu[i] = np.trace(sigma @ gamma_loc)

    if return_gamma:
        return nu, gamma
    else:
        return nu
    
    
def get_opt_exc_Pxx(freq, par_dict,
                    Pyy, Px_tot,
                    Pxx = None,
                    n_iter=3,
                    T_tot=None, n_avg=1,
                    dpar_dict=None, logflag_dict=None,
                    rec_progress=False):
    """
    Excitation optimization algorithm following Pintelon & Schoukens
    Sec. 5.4.2.2

    Use n_iter to control the numer of iterations to perform. 
    Typically n_iter=2 or 3 would be recommended, 
    otherwise the result would depend too heavily on the prior knowledge. 

    If just want the final PSD of the excitation, 
    set rec_progress=False.

    If want to examine how the fisher matrix & dispersion function changes,
    set rec_progress=True.
    """

    par, _, _ = unpack_par_dict(par_dict)

    n_par     = len(par)
    n_bin     = len(freq)
    Pxx_rec   = np.zeros([n_iter, n_bin])
    nu_rec    = np.zeros([n_iter, n_bin])
    gamma_rec = np.zeros([n_iter, n_par-1, n_par-1])

    # init
    # one can also start with a none-white initial excitation by passing an explicit Pxx. 
    if Pxx is None:
        Pxx = np.ones(n_bin)  
        
    Px_tot_temp = integ.trapezoid(Pxx, freq)
    Pxx *= Px_tot/Px_tot_temp

    cnt = 0
    while cnt < n_iter:
        nu, gamma = get_dispersion(freq, par_dict,
                                   Pxx, Pyy,
                                   T_tot=T_tot, n_avg=n_avg,
                                   dpar_dict=dpar_dict, logflag_dict=logflag_dict,
                                   return_gamma=True)

        Pxx = Pxx*nu
        Px_tot_temp = integ.trapezoid(Pxx, freq)
        Pxx *= Px_tot/Px_tot_temp

        Pxx_rec[cnt, :] = Pxx
        nu_rec[cnt, :] = nu
        gamma_rec[cnt, :, :] = gamma

        cnt += 1

    if rec_progress:
        return Pxx_rec, nu_rec, gamma_rec
    else:
        return Pxx 
    
########################################
###        TF estimation             ###
########################################

def invfreqs(w, H, wt, nb):
    """
    https://tspace.library.utoronto.ca/bitstream/1807/9981/1/Semlyen_9842_2889.pdf
    
    Note their b & a are opposite of the scipy convention
    They also go in the opposite order (increasing powers of w)
    Both are corrected to be consistent with scipy
    
    w is angular freq in [rad/Hz]
    H = H(w) is the TF evaluated at w
    
    wt is the weight. 
    If H_fit(w) = B(w)/A(w), the algoritm minimizes
        sum[ wt**2 |B(w) - A(w) H(w)|**2 ]
        
    We in general want 
        sum[ wt_desired**2 |B(w)/A(w) - H(w)|**2 ], 
    which means 
        wt = wt_desired / A(w)
    
    nb is the number of coefficients in the denominator - 1
    """
    
    pp = np.real(H)
    qq = np.imag(H)
    
    npt = len(pp)
    
    bb = np.zeros(2*npt)
    bb[0:2*npt:2] = pp
    bb[1:2*npt+1:2] = qq
    
    AA = np.zeros((2*npt, 2*nb))
    
    for i in range(npt):
        w_pow = w[i]**np.arange(nb+1)
        
        _sign = 1
        for j in range(0, nb-1, 2):
            
            AA[2*i, j] = w_pow[j] * _sign
            AA[2*i+1, j+1] = w_pow[j+1] * _sign
            
            AA[2*i,   j+nb]   =  qq[i] * w_pow[j+1] * _sign
            AA[2*i,   j+nb+1] =  pp[i] * w_pow[j+2] * _sign
            AA[2*i+1, j+nb]   = -pp[i] * w_pow[j+1] * _sign
            AA[2*i+1, j+nb+1] =  qq[i] * w_pow[j+2] * _sign
            
            _sign *= -1

    # weight; changing the residual
    ww = np.zeros(2*npt)
    ww[0:2*npt:2] = wt
    ww[1:2*npt+1:2] = wt
    ww = np.diag(ww)
    
    AA = np.dot(ww, AA)
    bb = np.dot(ww, bb)
    
    # rescaling; perserving residual
    inv_w_max = 1./np.max(w)
    inv_w_max_pow = inv_w_max**np.arange(nb)
    inv_H_max = 1./np.max(np.abs(H))
    
    DD = np.hstack((inv_w_max_pow, inv_w_max_pow * inv_w_max * inv_H_max))
    DD = np.diag(DD)
    AA = np.dot(AA, DD)
    
    # instead of inverting AA, use SVD
    UU, SS, VT = np.linalg.svd(AA)
    gg = np.dot(UU.T, bb)
    gg = gg[:2*nb]
    yy = gg/SS[:2*nb]
    xx = np.dot(VT.T, yy)
    
    # undo rescaling
    xx = np.dot(DD, xx)

    num = xx[:nb]
    den = np.hstack((1, xx[nb:]))
    return num[::-1], den[::-1] 


def update_par_dict_from_data(freq, G_data, G_err, par_dict0):
    """
    update the model based on new measurements. 
    
    For model give by:
            -----
        x ->| G |-> y
            -----        
    We have:
    G_err/ |G| = 1 / SNR
    """
    par, n_num, n_den = unpack_par_dict(par_dict0)
    
    den0 = par[n_num:]
    _, Gden = sig.freqs(np.array([1]), den0, worN=2.*np.pi*freq)
    
    df = np.gradient(freq)
    
    # 1/sigma of each freq bin
    # wt needs to be adjusted, see the comment in invfreqs
    wt = 1./(G_err * np.sqrt(df)) * np.abs(Gden)
    
    num, den = invfreqs(2.*np.pi*freq, G_data, wt, n_den-1)
    
    # restrict the number of terms in num to be the same as the model
    num = num[-n_num:]
    
    # rescale
    num /= den[-n_den]
    den /= den[-n_den]
    
    par_dict = {'num':num, 
                'den':den}
    return par_dict


def min_phi_FIR(numtaps, freq_in, H_mag_in, Nfft):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=840000
    
    freq_in needs to be normalized by fs
    """
    Nfft = _nextpow2(Nfft)
    ff = np.fft.fftfreq(Nfft)

    idx = freq_in>0
    min_log_H = np.log(np.min(H_mag_in))-10
    log_H_mag_vs_freq_func = interp.interp1d(np.log(freq_in[idx]), np.log(H_mag_in[idx]), \
                                 bounds_error=False, 
                                 fill_value=min_log_H)
    log_H = np.zeros(Nfft)
    log_H[1:] = log_H_mag_vs_freq_func(np.log(np.abs(ff[1:])))
    log_H[0] = min_log_H
    log_H[Nfft//2] = min_log_H
    
    win = np.hstack((
        sig.windows.tukey(Nfft//2, 0.01),sig.windows.tukey(Nfft//2, 0.01)))
    
    ss = np.sign(ff)
    ss[0], ss[Nfft//2] = 0, 0
    
    phi = np.real(-1j * np.fft.fft(
        ss * np.fft.ifft(log_H * win)))
    
    TF = (np.exp(log_H + 1j * phi))[:Nfft//2+1]
    TF *= sig.windows.tukey(len(TF), alpha=0.01)
    FIR = np.fft.irfft(TF)[:numtaps]
    return FIR


########################################
###     auxiliary functions          ###
########################################

def time_series_from_asd_vect(
        sec,
        fs,
        freq_in, asd_in,
        seed      = None, ):
    """
    Generate noise using tabulated ASD vector, asd_in, evaluated at frequency given by freq_in
    Parameters
    ----------
    sec : integer
        Length of time series in seconds.
    fs : integer
        Sampling frequency of time series in Hz. Defaults to 2048 Hz
    asd_file: path to the ASD file. The first column should be freq in [Hz] and the second one displacement in [m/rtHz]
    seed : int or np.random.default_rng(), optional
        Defaults to ``None``.
    Returns
    -------
    data : ndarray, shape (sec*fs,)
        Time series signal imitating LIGO noise.
    """

    if isinstance(seed, np.random._generator.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # generate the data at 2*sec, take the middle part to avoid boundaries
    Ndoub = int(2 * sec * fs)
    N = int(sec*fs)
    Nfft = _nextpow2(Ndoub)

    data = rng.standard_normal(Nfft) 

    freq = np.fft.rfftfreq(Nfft, d=1 / fs)
    data_fft = np.fft.rfft(data * sig.windows.tukey(Nfft, 0.2)) 
    df = freq[1]-freq[0]
    
    idx=np.where(freq_in>0)
    asd_vs_freq_func = interp.interp1d(freq_in[idx], np.log(asd_in[idx]), \
                                 bounds_error=False, fill_value=-np.inf)
    desired_asd = np.exp(asd_vs_freq_func(freq))

    # original asd ~ sqrt(2/fs)
    data_fft *= desired_asd * np.sqrt(fs/2.)
    
    data = np.fft.irfft(data_fft)
    # take the middle part to avoid boundary effects
    data = data[int(N/2):int(N/2)+N]
    return data


def _nextpow2(n):
    p = int(np.ceil(np.log2(n)))
    return 2**p

# convert from (f0, Q) to (a+1j*b, a-1j*b)
def get_res_g_pole_pair(f0, Q):
    """
    a, b are the real and imag parts of the complex pole
    """
    w0 = 2*np.pi * f0
    a  = -w0/2/Q
    b  = np.sqrt(w0**2 - a**2)

#     a=np.round(a, 6)
#     b=np.round(b, 6)

    return a, b


def get_res_f0_Q(a, b):
    """
    a, b are the real and imag parts of the complex pole
    """
    w0 = np.sqrt(a**2 + b**2)
    f0 = w0/2/np.pi
    Q  = -w0/2/a
    return f0, Q


# codes for plotting the error ellipses
def get_prob_contour(x0, w, v):
    """
    x0 is the true value (center of the ellipse)
    w is the eigenvalue  (semi-major/minor axes)
    v is the matrix formed by eigenvectors (for rotation)
    """
    nPt = 1000
    vT = v.transpose()
    y0 = np.dot(vT, x0)
    y1 = np.linspace(y0[0] - np.sqrt(w[0]), y0[0] + np.sqrt(w[0]), nPt)
    y2 = np.zeros([2, nPt], dtype=np.complex)
    y2[0, :] = y0[1]+np.sqrt(w[1])*np.sqrt(1.+0j-(y1-y0[0])**2./(w[0]))
    y2[1, :] = y0[1]-np.sqrt(w[1])*np.sqrt(1.+0j-(y1-y0[0])**2./(w[0]))

    x1 = np.zeros([2, nPt])
    x2 = np.zeros([2, nPt])
    for i in range(nPt):
        y_temp = np.array([y1[i], np.real(y2[0, i])])
        x_temp = np.dot(v, y_temp)
        x1[0, i] = x_temp[0]
        x2[0, i] = x_temp[1]

        y_temp = np.array([y1[i], np.real(y2[1, i])])
        x_temp = np.dot(v, y_temp)
        x1[1, i] = x_temp[0]
        x2[1, i] = x_temp[1]
    return x1, x2


def plot_prob_contour(sigma, theta, idx,
                      ax=None, label='', **ax_kwargs):
    """
    sigma is the covariance matrix
    theta is the list of true values
    idx is the indices for the two components whose covariance we want to check
    ax is an axis object onto which we plot the contour
    """
    err_mtrx = np.zeros([2, 2])
    err_mtrx[0, 0] = sigma[idx[0], idx[0]]
    err_mtrx[0, 1] = sigma[idx[0], idx[1]]
    err_mtrx[1, 0] = sigma[idx[1], idx[0]]
    err_mtrx[1, 1] = sigma[idx[1], idx[1]]
    w, v = np.linalg.eigh(err_mtrx)
    x1, x2 = get_prob_contour(np.array([theta[idx[0]], theta[idx[1]]]), w, v)
    if ax == None:
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
    ax.plot(x1[0, :], x2[0, :], label=label, **ax_kwargs)
    ax.plot(x1[1, :], x2[1, :], **ax_kwargs)
    return ax
    
    
    