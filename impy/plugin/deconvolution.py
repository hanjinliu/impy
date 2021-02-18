from flowdec.psf import GibsonLanni as PSF
from flowdec import restoration as fdres
from flowdec import data as fddata
import numpy as np
from ..func import record

__all__ = ["deconvolution3d"]

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

def _deconvolution3d(img3d, psfimg, niter) -> np.ndarray:
    img3d = np.array(img3d).astype("float64")

    if (img3d.shape != psfimg.shape):
        raise ValueError(f"shapes are inconsistent: {img3d.shape} and {psfimg.shape}")

    algorithm = fdres.RichardsonLucyDeconvolver(img3d.ndim).initialize()
    img_model = fddata.Acquisition(data=img3d, kernel=psfimg)
    result = algorithm.run(img_model, niter=niter)
    return result.data

@record
def deconvolution3d(self, psfinfo={}, niter=50):
    """
    Richardson-Lucy deconvolution.
    
    psfinfo:
    - For synthetic PSF image, wavelength [um], pxsize [um], dz [um] is needed.
        Default is ... wavelength=0.610, pxsize=0.216667, dz=0.38
    - For experimentally obtained image stack, use it directly. 
    
    niters: int
        number of iteration.
    """
    if (type(psfinfo) == dict):
        psfimg = psf(self.sizeof("x"), self.sizeof("y"), self.sizeof("z"), **psfinfo)
    else:
        psfimg = np.array(psfinfo)
        psfimg /= np.max(psfimg)

    out = np.zeros(self.shape)
    for t, img in self.as_uint16().iter("tc"):
        out[t] = _deconvolution3d(img, psfimg, niter)
    out = out.view(self.__class__)
    out._set_info(self, f"RL-Deconvolution(niter={niter})")
    
    return out.as_uint16()