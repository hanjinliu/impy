from skimage import morphology as skmorph
from skimage import transform as sktrans
from skimage import filters as skfil
from skimage import restoration as skres
from skimage import measure as skmes
from skimage.feature.corner import _symmetric_image
from skimage import feature as skfeat
from scipy import ndimage as ndi
import numpy as np
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft

def affine_(args):
    sl, data, mx, order = args
    return (sl, sktrans.warp(data, mx, order=order))

def median_(args):
    sl, data, selem = args
    return (sl, ndi.median_filter(data, footprint=selem, mode="reflect"))

def directional_median_(args):
    sl, data, radius = args
    kernels = directional_median_kernel_2d(radius*2 + 1)
    data_var = np.stack([ndi.convolve(data**2, ker/3, mode="reflect") - 
                         ndi.convolve(data, ker/3, mode="reflect")**2 for ker in kernels])
    min_vars = np.argmin(data_var, axis=0)
    data_med = [ndi.median_filter(data, footprint=ker, mode="reflect") for ker in kernels]
    out = np.empty_like(data)
    for d in [0, 1, 2, 3]:
        out = np.where(min_vars==d, data_med[d], out)
    
    return sl, out

def directional_median_kernel_2d(size):
    k1 = np.ones((1, size), dtype=np.uint8) # -
    k2 = np.eye(size)                       # \
    k3 = k1.T                               # |
    k4 = np.fliplr(k2)                      # /
    return [k1, k2, k3, k4]
    
def mean_(args):
    sl, data, selem = args
    return (sl, ndi.convolve(data, selem/np.sum(selem)))

def std_(args):
    sl, data, selem = args
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    std_img = np.sqrt(x2 - x1**2)
    return (sl, std_img)

def coef_(args):
    sl, data, selem = args
    selem = selem / np.sum(selem)
    x1 = ndi.convolve(data, selem)
    x2 = ndi.convolve(data**2, selem)
    # sometimes x2 is almost same as x1^2 and this causes negative value.
    out = np.sqrt(np.abs(x2 - x1**2))/x1
    return (sl, out)
    
def convolve_(args):
    sl, data, kernel, mode, cval = args
    return (sl, ndi.convolve(data, kernel, mode=mode, cval=cval))

def gaussian_(args):
    sl, data, sigma = args
    return (sl, ndi.gaussian_filter(data, sigma))

def entropy_(args):
    sl, data, selem = args
    return (sl, skfil.rank.entropy(data, selem))

def enhance_contrast_(args):
    sl, data, selem = args
    return (sl, skfil.rank.enhance_contrast(data, selem))

def difference_of_gaussian_(args):
    sl, data, low_sigma, high_sigma = args
    return (sl, skfil.difference_of_gaussians(data, low_sigma, high_sigma))

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
    return (sl, det)

def gaussian_laplace_(args):
    sl, data, sigma = args
    return (sl, ndi.gaussian_laplace(data, sigma))

def rolling_ball_(args):
    sl, data, radius, smooth = args
    if smooth:
        _, ref = mean_((sl, data, np.ones((3, 3))))
        back = skres.rolling_ball(ref, radius=radius)
        tozero = (back > data)
        back[tozero] = data[tozero]
    else:
        back = skres.rolling_ball(data, radius=radius)
    
    return (sl, data - back)

def sobel_(args):
    sl, data = args
    return (sl, skfil.sobel(data))
    
def opening_(args):
    sl, data, selem = args
    return (sl, skmorph.opening(data, selem))

def binary_opening_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_opening(data, selem))

def closing_(args):
    sl, data, selem = args
    return (sl, skmorph.closing(data, selem))

def binary_closing_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_closing(data, selem))

def erosion_(args):
    sl, data, selem = args
    return (sl, skmorph.erosion(data, selem))

def binary_erosion_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_erosion(data, selem))

def dilation_(args):
    sl, data, selem = args
    return (sl, skmorph.dilation(data, selem))

def binary_dilation_(args):
    sl, data, selem = args
    return (sl, skmorph.binary_dilation(data, selem))

def tophat_(args):
    sl, data, selem = args
    return (sl, skmorph.white_tophat(data, selem))

def convex_hull_(args):
    sl ,data = args
    return (sl, skmorph.convex_hull_image(data))

def skeletonize_(args):
    sl, data = args
    return (sl, skmorph.skeletonize_3d(data))

def hessian_eigh_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(hessian)
    return (sl, eigval, eigvec)

def hessian_eigval_(args):
    sl, data, sigma, pxsize = args
    hessian_elements = skfeat.hessian_matrix(data, sigma=sigma, order="xy",
                                             mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    hessian = _symmetric_image(hessian_elements)
    hessian *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(hessian)
    return (sl, eigval)

def structure_tensor_eigh_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval, eigvec = np.linalg.eigh(tensor)
    return (sl, eigval, eigvec)

def structure_tensor_eigval_(args):
    sl, data, sigma, pxsize = args
    tensor_elements = skfeat.structure_tensor(data, sigma, order="xy",
                                              mode="reflect")
    # Correct for scale
    pxsize = np.asarray(pxsize)
    tensor = _symmetric_image(tensor_elements)
    tensor *= (pxsize.reshape(-1,1) * pxsize.reshape(1,-1))
    eigval = np.linalg.eigvalsh(tensor)
    return (sl, eigval)

def label_(args):
    sl, data, connectivity = args
    labels = skmes.label(data, background=0, connectivity=connectivity)
    return (sl, labels)

def distance_transform_edt_(args):
    sl, data = args
    return (sl, ndi.distance_transform_edt(data))

def fill_hole_(args):
    sl, data, mask = args
    seed = np.copy(data)
    seed[1:-1, 1:-1] = data.max()
    return (sl, skmorph.reconstruction(seed, mask, method="erosion"))

def count_neighbors_(args):
    sl, data, selem = args
    return (sl, ndi.convolve(data.astype("uint8"), selem, mode="constant") - data)

def corner_harris_(args):
    sl, data, k, sigma = args
    return (sl, skfeat.corner_harris(data, k=k, sigma=sigma))

def wiener_(args):
    sl, obs, psf_ft, psf_ft_conj, lmd = args
    
    img_ft = fft(obs)
    
    estimated = np.real(ifft(img_ft*psf_ft_conj / (psf_ft*psf_ft_conj + lmd)))
    return sl, np.fft.fftshift(estimated)
    
def richardson_lucy_(args):
    # Identical to the algorithm in Deconvolution.jl of Julia.
    sl, obs, psf_ft, psf_ft_conj, niter = args
    
    def lucy_step(estimated):
        factor = ifft(fft(obs / ifft(fft(estimated) * psf_ft)) * psf_ft_conj)
        return estimated * np.real(factor)
    
    estimated = np.real(ifft(fft(obs) * psf_ft))
    for _ in range(niter):
        estimated = lucy_step(estimated)
    
    return sl, np.fft.fftshift(estimated)

def estimate_sigma_(args):
    sl, data = args
    return (sl[:-data.ndim], skres.estimate_sigma(data))