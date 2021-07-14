import numpy as np
from functools import partial
from scipy.fft import rfftn as rfft
from scipy.fft import irfftn as irfft

def wiener(obs, psf_ft, psf_ft_conj, lmd):
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    
    img_ft = fft(obs)
    
    estimated = np.real(ifft(img_ft*psf_ft_conj / (psf_ft*psf_ft_conj + lmd)))
    return np.fft.fftshift(estimated)
    
def richardson_lucy(obs, psf_ft, psf_ft_conj, niter, eps):
    # Identical to the algorithm in Deconvolution.jl of Julia.
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    conv = factor = np.empty(obs.shape, dtype=np.float32) # placeholder
    estimated = np.real(ifft(fft(obs) * psf_ft))   # initialization
    
    for _ in range(niter):
        conv[:] = ifft(fft(estimated) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        estimated *= factor
        
    return np.fft.fftshift(estimated)

def richardson_lucy_tv(obs, psf_ft, psf_ft_conj, max_iter, lmd, tol, eps):
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    est_old = ifft(fft(obs) * psf_ft).real
    est_new = np.empty(obs.shape, dtype=np.float32)
    conv = factor = norm = gg = np.empty(obs.shape, dtype=np.float32) # placeholder
    
    for _ in range(max_iter):
        conv[:] = ifft(fft(est_old) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        est_new[:] = est_old * factor
        grad = np.gradient(est_old)
        norm[:] = np.sqrt(sum(g**2 for g in grad))
        gg[:] = sum(np.gradient(_safe_div(grad[i], norm, eps=1e-8), axis=i) 
                    for i in range(obs.ndim))
        est_new /= (1 - lmd * gg)
        gain = np.sum(np.abs(est_new - est_old))/np.sum(np.abs(est_old))
        if gain < tol:
            break
        est_old[:] = est_new
        
    return np.fft.fftshift(est_new)


def _safe_div(a, b, eps=1e-8):
    out = np.zeros(a.shape, dtype=np.float32)
    mask = b > eps
    out[mask] = a[mask]/b[mask]
    return out