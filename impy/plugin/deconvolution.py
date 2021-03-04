from flowdec.psf import GibsonLanni as PSF
from flowdec import restoration as fdres
from flowdec import data as fddata
import numpy as np
from ..func import record

__all__ = ["deconvolute_cf", "deconvolute_tirf"]

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

def _rechardson_lucy(img, psfimg, deconvinfo, niter) -> np.ndarray:
    img = np.array(img).astype("float32")
    kw = {"epsilon": np.percentile(img, 99)*1e-6}
    kw.update(deconvinfo)
    
    algorithm = fdres.RichardsonLucyDeconvolver(img.ndim, **kw).initialize()
    img_model = fddata.Acquisition(data=img, kernel=psfimg)
    result = algorithm.run(img_model, niter=niter)
    return result.data

@record
def deconvolute_cf(self, psfinfo={}, deconvinfo={}, niter:int=50):
    """
    Deconvolution of 3-dimensional image obtained from confocal microscopy, 
    using Richardson-Lucy's algorithm.
    
    Parameters
    ----------
    psfinfo: dict or np.array
        For synthetic PSF image, pass dict of PSF parameters. By default, 
        wavelength=0.610, pxsize=0.216667, dz=0.38. For experimentally obtained
        PSF image stack, use it directly.
    
    deconvinfo: dict
        Keyword arguments passed to RichardsonLucyDeconvolver.

    niters: int
        Number of iteration.
    """
    if (isinstance(psfinfo, dict)):
        # 7x7 is enough for standard conditions.
        # set size_z to an odd number
        kw = {"size_x": 7, "size_y": 7, "size_z": self.sizeof("z")//2*2+1}
        kw.update(psfinfo)
        psfimg = psf(**kw)
    elif (isinstance(psfinfo, np.ndarray)):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")

    out = np.zeros(self.shape)
    for sl, img in self.iter("tcs"):
        out[sl] = _rechardson_lucy(img, psfimg, deconvinfo, niter)
    out = out.view(self.__class__)
    out._set_info(self, f"RichardsonLucy-confocal(niter={niter})")
    
    return out.as_uint16()

@record
def deconvolute_tirf(self, psfinfo={}, deconvinfo={}, niter:int=50):
    if (isinstance(psfinfo, dict)):
        # 7x7 is enough for standard conditions.
        kw = {"size_x": self.sizeof("x"), "size_y": self.sizeof("y"), "size_z": 1}
        kw.update(psfinfo)
        psfimg = psf(**kw)[0]
    elif (isinstance(psfinfo, np.ndarray)):
        psfimg = np.asarray(psfinfo)
        psfimg /= np.max(psfimg)
    else:
        raise TypeError(f"'psfinfo' must be dict or np.ndarray, but got {type(psfinfo)}")

    out = np.zeros(self.shape)
    for sl, img in self.iter("tzcs"):
        out[sl] = _rechardson_lucy(img, psfimg, deconvinfo, niter)
    out = out.view(self.__class__)
    out._set_info(self, f"RichardsonLucy-TIRF(niter={niter})")
    
    return out.as_uint16()