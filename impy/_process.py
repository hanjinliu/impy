from ._skimage import *
from skimage.feature.corner import _symmetric_image
import numpy as np
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft

def affine_(args):
    sl, data, mx, order = args
    return sl, sktrans.warp(data, mx, order=order)

def median_(args):
    sl, data, selem = args
    return sl, ndi.median_filter(data, footprint=selem, mode="reflect")

def directional_median_(args):
    sl, data, radius = args
    diam = 2*radius + 1
    directional_median_kernel = directional_median_kernel_2d if data.ndim == 2 else directional_median_kernel_3d
    kernels = directional_median_kernel(radius*2 + 1)
    data_var = np.stack([ndi.convolve(data**2, ker/diam, mode="reflect") - 
                         ndi.convolve(data, ker/diam, mode="reflect")**2 for ker in kernels])
    min_vars = np.argmin(data_var, axis=0)
    data_med = [ndi.median_filter(data, footprint=ker, mode="reflect") for ker in kernels]
    out = np.empty_like(data)
    for d in np.arange(len(kernels)):
        out = np.where(min_vars==d, data_med[d], out)
    
    return sl, out

def directional_median_kernel_2d(size):
    k1 = np.ones((1, size), dtype=np.uint8) # -
    k2 = np.eye(size)                       # \
    k3 = k1.T                               # |
    k4 = np.fliplr(k2)                      # /
    return [k1, k2, k3, k4]

def directional_median_kernel_3d(size):
    raise NotImplementedError
    
def wavelet_denoising_(args):
    sl, data, func_kw, max_shift, shift_steps = args
    out = skres.cycle_spin(data, skres.denoise_wavelet, func_kw=func_kw, max_shifts=max_shift, 
                           multichannel=False, shift_steps=shift_steps)
    return sl, out
    
def mean_(args):
    sl, data, selem = args
    return sl, ndi.convolve(data, selem/np.sum(selem))

def phase_mean_(args):
    sl, data, selem, a = args
    out = np.empty_like(data, dtype=np.complex64)
    np.exp(1j*a*data, out=out)
    ndi.convolve(out, selem, output=out)
    return sl, np.angle(out)/a
    
def std_(args):
    sl, data, selem = args
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    std_img = np.sqrt(x2 - x1**2)
    return sl, std_img

def coef_(args):
    sl, data, selem = args
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    # sometimes x2 is almost same as x1^2 and this causes negative value.
    out = np.sqrt(np.abs(x2 - x1**2))/x1
    return sl, out
    
def convolve_(args):
    sl, data, kernel, mode, cval = args
    return sl, ndi.convolve(data, kernel, mode=mode, cval=cval)

def gaussian_(args):
    sl, data, sigma = args
    return sl, ndi.gaussian_filter(data, sigma)

def entropy_(args):
    sl, data, selem = args
    return sl, skfil.rank.entropy(data, selem)

def enhance_contrast_(args):
    sl, data, selem = args
    return sl, skfil.rank.enhance_contrast(data, selem)

def difference_of_gaussian_(args):
    sl, data, low_sigma, high_sigma = args
    return sl, skfil.difference_of_gaussians(data, low_sigma, high_sigma)

def population_(args):
    sl, data, selem = args
    return sl, skfil.rank.pop(data, selem, mask=data)

def hessian_det_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    eigval[eigval>0] = 0
    det = np.product(eigval, axis=-1)
    return sl, det

def gaussian_laplace_(args):
    sl, data, sigma = args
    return sl, ndi.gaussian_laplace(data, sigma)

def rolling_ball_(args):
    sl, data, radius, prefilter = args
    if prefilter == "mean":
        _, data = mean_((sl, data, np.ones((3, 3))))
    elif prefilter == "median":
        _, data = median_((sl, data, np.ones((3, 3))))
    
    back = skres.rolling_ball(data, radius=radius)
    
    return sl, back

def rof_filter_(args):
    sl, obs, lmd, tol, max_iter = args
    out = skres.denoise_tv_chambolle(obs, weight=lmd, eps=tol, n_iter_max=max_iter)
    return sl, out

def sobel_(args):
    sl, data = args
    return sl, skfil.sobel(data)

def farid_(args):
    sl, data = args
    return sl, skfil.farid(data)

def scharr_(args):
    sl, data = args
    return sl, skfil.scharr(data)

def prewitt_(args):
    sl, data = args
    return sl, skfil.prewitt(data)

def sobel_h_(args):
    sl, data = args
    return sl, skfil.sobel_h(data)

def sobel_v_(args):
    sl, data = args
    return sl, skfil.sobel_v(data)

def farid_h_(args):
    sl, data = args
    return sl, skfil.farid_h(data)

def farid_v_(args):
    sl, data = args
    return sl, skfil.farid_v(data)

def scharr_h_(args):
    sl, data = args
    return sl, skfil.scharr_h(data)

def scharr_v_(args):
    sl, data = args
    return sl, skfil.scharr_v(data)

def prewitt_h_(args):
    sl, data = args
    return sl, skfil.prewitt_h(data)

def prewitt_v_(args):
    sl, data = args
    return sl, skfil.prewitt_v(data)

def opening_(args):
    sl, data, selem = args
    return sl, skmorph.opening(data, selem)

def binary_opening_(args):
    sl, data, selem = args
    return sl, skmorph.binary_opening(data, selem)

def closing_(args):
    sl, data, selem = args
    return sl, skmorph.closing(data, selem)

def binary_closing_(args):
    sl, data, selem = args
    return sl, skmorph.binary_closing(data, selem)

def erosion_(args):
    sl, data, selem = args
    return sl, skmorph.erosion(data, selem)

def binary_erosion_(args):
    sl, data, selem = args
    return sl, skmorph.binary_erosion(data, selem)

def dilation_(args):
    sl, data, selem = args
    return sl, skmorph.dilation(data, selem)

def binary_dilation_(args):
    sl, data, selem = args
    return sl, skmorph.binary_dilation(data, selem)

def diameter_opening_(args):
    sl, data, diam, connectivity = args
    return sl, skmorph.diameter_opening(data, diam, connectivity)

def diameter_closing_(args):
    sl, data, diam, connectivity = args
    return sl, skmorph.diameter_closing(data, diam, connectivity)

def area_opening_(args):
    sl, data, area, connectivity = args
    return sl, skmorph.area_opening(data, area, connectivity)

def binary_area_opening_(args):
    sl, data, area, connectivity = args
    return sl, skmorph.remove_small_objects(data, area, connectivity)

def area_closing_(args):
    sl, data, area, connectivity = args
    return sl, skmorph.area_closing(data, area, connectivity)

def binary_area_closing_(args):
    sl, data, area, connectivity = args
    return sl, skmorph.remove_small_holes(data, area, connectivity)

def tophat_(args):
    sl, data, selem = args
    return sl, skmorph.white_tophat(data, selem)

def convex_hull_(args):
    sl ,data = args
    return sl, skmorph.convex_hull_image(data)

def skeletonize_(args):
    sl, data, selem = args
    skl = skmorph.skeletonize_3d(data)
    if selem is not None:
        skl = skmorph.binary_dilation(skl, selem)
    return sl, skl 

def hessian_eigh_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return sl, eigval, eigvec

def hessian_eigval_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return sl, eigval

def structure_tensor_eigh_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return sl, eigval, eigvec

def structure_tensor_eigval_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return sl, eigval

def gabor_(args):
    sl, data, ker = args
    out = np.empty_like(data, dtype=np.complex64)
    out.real[:] = ndi.convolve(data, ker.real)
    out.imag[:] = ndi.convolve(data, ker.imag)
    return sl, out

def gabor_real_(args):
    sl, data, ker = args
    out = ndi.convolve(data, ker.real)
    return sl, out

def lbp_(args):
    sl, data, p, radius, method = args
    out = skfeat.local_binary_pattern(data, p, radius, method)
    return sl, out

def glcm_(args):
    sl, data, distances, angles, levels = args
    out = skfeat.greycomatrix(data, distances, angles, levels=levels)
    return sl, out

def label_(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return sl, labels

def distance_transform_edt_(args):
    sl, data = args
    return sl, ndi.distance_transform_edt(data)

def fill_hole_(args):
    sl, data, mask = args
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.max()
    return sl, skmorph.reconstruction(seed, mask, method="erosion")

def corner_harris_(args):
    sl, data, k, sigma = args
    return sl, skfeat.corner_harris(data, k=k, sigma=sigma)

def wiener_(args):
    sl, obs, psf_ft, psf_ft_conj, lmd = args
    
    img_ft = fft(obs)
    
    estimated = np.real(ifft(img_ft*psf_ft_conj / (psf_ft*psf_ft_conj + lmd)))
    return sl, np.fft.fftshift(estimated)
    
def richardson_lucy_(args):
    # Identical to the algorithm in Deconvolution.jl of Julia.
    sl, obs, psf_ft, psf_ft_conj, niter = args
    
    factor = np.empty(obs.shape, dtype=np.float32) # placeholder
    estimated = np.real(ifft(fft(obs) * psf_ft))   # initialization
    
    for _ in range(niter):
        factor[:] = ifft(fft(obs / ifft(fft(estimated) * psf_ft)) * psf_ft_conj).real
        estimated *= factor
        
    return sl, np.fft.fftshift(estimated)

def richardson_lucy_tv_(args):
    sl, obs, psf_ft, psf_ft_conj, max_iter, lmd, tol = args
    
    est_old = ifft(fft(obs) * psf_ft).real
    est_new = np.empty(obs.shape, dtype=np.float32)
    factor = norm = gg = np.empty(obs.shape, dtype=np.float32) # placeholder
    
    with np.errstate(all="ignore"):
        # NOTE: During iteration, sometimes the line `gg[:] = ...` returns RuntimeWarning due to
        # unknown reasons. I checked with np.isfinite but could not find anything wrong. I set
        # error state to `all="ingore"`` for now as a quick solution.
        for _ in range(max_iter):
            factor[:] = ifft(fft(obs / ifft(fft(est_old) * psf_ft)) * psf_ft_conj).real
            est_new[:] = est_old * factor
            grad = np.gradient(est_old)
            norm[:] = np.sqrt(sum(g**2 for g in grad))
            # TODO: do not use np.gradient
            gg[:] = sum(np.gradient(np.where(norm<1e-8, 0, grad[i]/norm), axis=i) 
                        for i in range(obs.ndim))
            est_new /= (1 - lmd * gg)
            gain = np.sum(np.abs(est_new - est_old))/np.sum(np.abs(est_old))
            
            if gain < tol:
                break
            est_old[:] = est_new
        
    return sl, np.fft.fftshift(est_new)
