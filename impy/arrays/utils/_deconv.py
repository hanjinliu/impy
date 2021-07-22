from functools import partial
try:
    import cupy as np
    from cupyx.scipy.fft import rfftn as rfft
    from cupyx.scipy.fft import irfftn as irfft
    fftshift = lambda x: np.fft.fftshift(x).get()
    # CUDA <= ver.8 does not have gradient
    try:
        gradient = np.gradient
    except AttributeError:
        import numpy
        def gradient(a, axis=None):
            out = numpy.gradient(a.get(), axis=axis)
            return np.asarray(out)

except ImportError:
    import numpy as np
    from scipy.fft import rfftn as rfft
    from scipy.fft import irfftn as irfft
    fftshift = np.fft.fftshift
    gradient = np.gradient

__all__ = ["wiener", "lucy", "lucy_tv", "check_psf"]

def wiener(obs, psf_ft, psf_ft_conj, lmd):
    obs = np.asarray(obs)
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    
    img_ft = fft(obs)
    
    estimated = np.real(ifft(img_ft*psf_ft_conj / (psf_ft*psf_ft_conj + lmd)))
    return fftshift(estimated)
    
def richardson_lucy(obs, psf_ft, psf_ft_conj, niter, eps):
    # Identical to the algorithm in Deconvolution.jl of Julia.
    obs = np.asarray(obs)
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    conv = factor = np.empty(obs.shape, dtype=np.float32) # placeholder
    estimated = np.real(ifft(fft(obs) * psf_ft))   # initialization
    
    for _ in range(niter):
        conv[:] = ifft(fft(estimated) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        estimated *= factor
        
    return fftshift(estimated)

def richardson_lucy_tv(obs, psf_ft, psf_ft_conj, max_iter, lmd, tol, eps):
    obs = np.asarray(obs)
    fft = rfft
    ifft = partial(irfft, s=obs.shape)
    est_old = ifft(fft(obs) * psf_ft).real
    est_new = np.empty(obs.shape, dtype=np.float32)
    conv = factor = norm = gg = np.empty(obs.shape, dtype=np.float32) # placeholder
    
    for _ in range(max_iter):
        conv[:] = ifft(fft(est_old) * psf_ft).real
        factor[:] = ifft(fft(_safe_div(obs, conv, eps=eps)) * psf_ft_conj).real
        est_new[:] = est_old * factor
        grad = gradient(est_old)
        norm[:] = np.sqrt(sum(g**2 for g in grad))
        gg[:] = sum(gradient(_safe_div(grad[i], norm, eps=1e-8), axis=i) 
                    for i in range(obs.ndim))
        est_new /= (1 - lmd * gg)
        gain = np.sum(np.abs(est_new - est_old))/np.sum(np.abs(est_old))
        if gain < tol:
            break
        est_old[:] = est_new
        
    return fftshift(est_new)


def _safe_div(a, b, eps=1e-8):
    out = np.zeros(a.shape, dtype=np.float32)
    mask = b > eps
    out[mask] = a[mask]/b[mask]
    return out


def check_psf(img, psf, dims):
    psf = np.asarray(psf, dtype=np.float32)
    psf /= np.sum(psf)
    
    if img.sizesof(dims) != psf.shape:
        raise ValueError("observation and PSF have different shape: "
                        f"{img.sizesof(dims)} and {psf.shape}")
    psf_ft = rfft(psf)
    psf_ft_conj = np.conjugate(psf_ft)
    return psf_ft, psf_ft_conj