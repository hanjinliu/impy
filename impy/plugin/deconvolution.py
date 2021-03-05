from flowdec.psf import GibsonLanni as PSF
import numpy as np
from scipy.fftpack import fftn as fft
from scipy.fftpack import ifftn as ifft
from ..func import record

# To avoid Memory Error, scipy.fftpack is used instead of numpy.fft because the latter does not support 
# dtype complex64.

__all__ = ["lucy3d", "lucy2d"]

def psf(size_x, size_y, size_z, wavelength:float=0.610, pxsize:float=0.216667, dz:float=0.38, **kwargs) -> np.ndarray:
    """
    wavelength: wave length [micron]
    pxsize: pixel size [micron]
    dz: slice interval [micron]
    ns: reflactive index of the specimen
    na: N.A.
    """
    
    # in case parameters were set in nm-unit
    if (wavelength > 200):
        raise ValueError(f"'wavelength' is too large: {wavelength} um")
    if (pxsize > 50):
        raise ValueError(f"'pxsize' is too large: {pxsize} um")
    if (dz > 10):
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

# def _rechardson_lucy(img, psfimg, deconvinfo, niter) -> np.ndarray:
#     img = np.array(img).astype("float32")
#     kw = {"epsilon": np.percentile(img, 99)*1e-6}
#     kw.update(deconvinfo)
    
#     algorithm = fdres.RichardsonLucyDeconvolver(img.ndim, **kw).initialize()
#     img_model = fddata.Acquisition(data=img, kernel=psfimg)
#     result = algorithm.run(img_model, niter=niter)
#     return result.data

def _richardson_lucy(args):
    sl, obs, psf, niter = args
    
    if (obs.shape != psf.shape):
        raise ValueError(f"observation and PSF have different shape: {obs.shape} and {psf.shape}")
    
    obs = obs.astype("float32")
    psf = psf.astype("float32")
    
    psf_ft = fft(psf)
    psf_ft_conj = np.conjugate(psf_ft)
    
    def lucy_step(estimated):
        factor = ifft(fft(obs / ifft(fft(estimated) * psf_ft)) * psf_ft_conj)
        return estimated * np.real(factor)
    
    estimated = np.real(ifft(fft(obs) * psf_ft))
    for _ in range(niter):
        estimated = lucy_step(estimated)
    
    return sl, np.fft.fftshift(estimated)

@record
def lucy3d(self, psfinfo={}, niter:int=50, dtype="uint16", n_cpu=4):
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
    if (isinstance(psfinfo, dict)):
        kw = {"size_x": self.sizeof("x"), "size_y": self.sizeof("y"), "size_z": self.sizeof("z")}
        kw.update(psfinfo)
        psfimg = psf(**kw)
    elif (isinstance(psfinfo, np.ndarray)):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")
    
    # start deconvolution
    out = np.zeros(self.shape)
    out = self.parallel(_richardson_lucy, "ptcs", psfimg, niter, n_cpu=n_cpu)
    out = out.view(self.__class__)
    out._set_info(self, f"RichardsonLucy-3D(niter={niter})")
    
    return out.as_img_type(dtype)

@record
def lucy2d(self, psfinfo={}, niter:int=50, dtype="uint16", n_cpu=4):
    # make PSF
    if (isinstance(psfinfo, dict)):
        kw = {"size_x": self.sizeof("x"), "size_y": self.sizeof("y"), "size_z": 1}
        kw.update(psfinfo)
        psfimg = psf(**kw)[0]
    elif (isinstance(psfinfo, np.ndarray)):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")
    
    out = np.zeros(self.shape)
    out = self.parallel(_richardson_lucy, "ptzcs", psfimg, niter, n_cpu=n_cpu)
    out = out.view(self.__class__)
    out._set_info(self, f"RichardsonLucy-2D(niter={niter})")
    
    return out.as_img_type(dtype)
