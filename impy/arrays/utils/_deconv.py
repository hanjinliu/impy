from functools import partial
from ..._cupy import xp, xp_fft, asnumpy

try:
    gradient = xp.gradient
except AttributeError:
    # CUDA <= ver.8 does not have gradient
    import numpy
    def gradient(a, axis=None):
        out = numpy.gradient(a.get(), axis=axis)
        return xp.asarray(out)

__all__ = ["wiener", 
           "lucy",
           "lucy_tv",
           "check_psf",
           ]

def wiener(obs, psf_ft, psf_ft_conj, lmd):
    obs = xp.asarray(obs)
    fft = xp_fft.rfftn
    ifft = partial(xp_fft.irfftn, s=obs.shape)
    
    img_ft = fft(obs)
    
    estimated = xp.real(ifft(img_ft*psf_ft_conj / (psf_ft*psf_ft_conj + lmd)))
    return asnumpy(xp_fft.fftshift(estimated))
    
def richardson_lucy(obs, psf_ft, psf_ft_conj, niter, eps):
    # Identical to the algorithm in Deconvolution.jl of Julia.
    obs = xp.asarray(obs)
    fft = xp_fft.rfftn
    ifft = partial(xp_fft.irfftn, s=obs.shape)
    conv = factor = xp.empty(obs.shape, dtype=xp.float32) # placeholder
    estimated = xp.real(ifft(fft(obs) * psf_ft))   # initialization
    
    for _ in range(niter):
        conv[:] = ifft(fft(estimated) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        estimated *= factor
        
    return asnumpy(xp_fft.fftshift(estimated))

def richardson_lucy_tv(obs, psf_ft, psf_ft_conj, max_iter, lmd, tol, eps):
    obs = xp.asarray(obs)
    fft = xp_fft.rfftn
    ifft = partial(xp_fft.irfftn, s=obs.shape)
    est_old = ifft(fft(obs) * psf_ft).real
    est_new = xp.empty(obs.shape, dtype=xp.float32)
    conv = factor = norm = gg = xp.empty(obs.shape, dtype=xp.float32) # placeholder
    
    for _ in range(max_iter):
        conv[:] = ifft(fft(est_old) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        est_new[:] = est_old * factor
        grad = gradient(est_old)
        norm[:] = xp.sqrt(sum(g**2 for g in grad))
        gg[:] = sum(gradient(_safe_div(grad[i], norm, eps=1e-8), axis=i) 
                    for i in range(obs.ndim))
        est_new /= (1 - lmd * gg)
        gain = xp.sum(xp.abs(est_new - est_old))/xp.sum(xp.abs(est_old))
        if gain < tol:
            break
        est_old[:] = est_new
        
    return asnumpy(xp_fft.fftshift(est_new))


def _safe_div(a, b, eps=1e-8):
    out = xp.zeros(a.shape, dtype=xp.float32)
    mask = b > eps
    out[mask] = a[mask]/b[mask]
    return out


def check_psf(img, psf, dims):
    psf = xp.asarray(psf, dtype=xp.float32)
    psf /= xp.sum(psf)
    
    if img.sizesof(dims) != psf.shape:
        raise ValueError("observation and PSF have different shape: "
                        f"{img.sizesof(dims)} and {psf.shape}")
    psf_ft = xp_fft.rfftn(psf)
    psf_ft_conj = xp.conjugate(psf_ft)
    return psf_ft, psf_ft_conj