import numpy as np
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft
from ..deco import *

# Identical to the algorithm in Deconvolution.jl of Julia.
# To avoid Memory Error, scipy.fftpack is used instead of numpy.fft because the latter does not support 
# dtype complex64.

__all__ = ["lucy3d", "lucy2d"]

def synthesize_psf(size_x, size_y, size_z, wavelength:float=0.610, 
        pxsize:float=0.216667, dz:float=0.38, **kwargs) -> np.ndarray:
    """
    wavelength: wave length [micron]
    pxsize: pixel size [micron]
    dz: slice interval [micron]
    ns: reflactive index of the specimen
    na: N.A.
    """
    try:
        from flowdec.psf import GibsonLanni as PSF
    except ImportError:
        raise ImportError("You need to install flowdec to make "
                          "synthetic point spread function")
    
    # in case parameters were set in nm-unit
    if wavelength > 200:
        raise ValueError(f"'wavelength' is too large: {wavelength} um")
    if pxsize > 50:
        raise ValueError(f"'pxsize' is too large: {pxsize} um")
    if dz > 10:
        raise ValueError(f"'dz' is too large: {dz} um")
    
    # set parameters
    psf_kwargs = {"size_x":size_x, "size_y":size_y, "size_z":size_z,
                  "wavelength":wavelength, "res_lateral":pxsize, "res_axial":dz}
    # Olympus, PlanApoN 60x/1.40 Oil and Matsunami glass
    psf_kwargs.update({"m":60, "ti0":150, "ns":1.518, "na":1.4, "tg0":140, "ng0":1.526})

    # set other parameters
    psf_kwargs.update(kwargs)
    
    # make PSF (z,y,x-order)
    psfimg = PSF(**psf_kwargs).generate()
    return psfimg

def _richardson_lucy(args):
    sl, obs, psf, niter = args
    # obs and psf must be float32 here
    
    if obs.shape != psf.shape:
        raise ValueError("observation and PSF have different shape: "
                        f"{obs.shape} and {psf.shape}")
    
    psf_ft = fft(psf)
    psf_ft_conj = np.conjugate(psf_ft)
    
    def lucy_step(estimated):
        factor = ifft(fft(obs / ifft(fft(estimated) * psf_ft)) * psf_ft_conj)
        return estimated * np.real(factor)
    
    estimated = np.real(ifft(fft(obs) * psf_ft))
    for _ in range(niter):
        estimated = lucy_step(estimated)
    
    return sl, np.fft.fftshift(estimated)

@same_dtype(asfloat=True)
@record()
def lucy3d(self, psfinfo, niter:int=50, update:bool=False):
    """
    Deconvolution of 3-dimensional image obtained from confocal microscopy, 
    using Richardson-Lucy's algorithm.
    
    Parameters
    ----------
    psfinfo : dict or np.array
        For synthetic PSF image, pass dict of PSF parameters. By default, 
        wavelength=0.610, pxsize=0.216667, dz=0.38. For experimentally obtained
        PSF image stack, use it directly.

    niters : int
        Number of iteration.
    
    dtype : str
        Output dtype
    """
    # make PSF
    if isinstance(psfinfo, dict):
        kw = {"size_x": self.sizeof("x"), "size_y": self.sizeof("y"), "size_z": self.sizeof("z")}
        kw.update(psfinfo)
        psfimg = synthesize_psf(**kw)
    elif isinstance(psfinfo, np.ndarray):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")
    
    psfimg = psfimg.astype("float32")
    
    # start deconvolution
    out = np.zeros(self.shape)
    out = self.parallel(_richardson_lucy, "ptc", psfimg, niter)
    
    return out

@same_dtype(asfloat=True)
@record()
def lucy2d(self, psfinfo, niter:int=50, update:bool=False):
    # make PSF
    if isinstance(psfinfo, dict):
        kw = {"size_x": self.sizeof("x"), "size_y": self.sizeof("y"), "size_z": 1}
        kw.update(psfinfo)
        psfimg = synthesize_psf(**kw)[0]
    elif (isinstance(psfinfo, np.ndarray)):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")
    
    psfimg = psfimg.astype("float32")
    
    # start deconvolution
    out = np.zeros(self.shape)
    out = self.parallel(_richardson_lucy, "ptzc", psfimg, niter)
    
    return out
