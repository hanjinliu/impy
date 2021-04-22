__version__ = "1.4.5"
# TODO: btrack ... https://github.com/quantumjot/BayesianTracker

import warnings
from .imgarray import (array, zeros, zeros_like, empty, empty_like, 
                       imread, imread_collection, read_meta, 
                       stack, set_cpu, ImgArray)
from .specials import PropArray, MarkerArray

try:
    from .viewer import window
except ImportError as e:
    print(f"Could not import viewer: {e}")
except Exception as e:
    print(f"Could not import viewer: {e}")

__doc__ = \
r"""
Extended version of NumPy for image analysis.
The name of original image and executed calculations are all recorded.
(See also __array_finalize__, __array_ufunc__ and the NumPy official documents)

Examples
--------
Import module
>>> import impy as ip
Load tiff image by
>>> img = ip.imread(r"C:\Users\...\xxx.tif")
or directly converting numpy ndarray by
>>> img = ip.array(A)

(ex.1) Fitting background to 2-D Gaussian
Median filtering.
>>> img_med = img.median_filter(radius=30)
Fitting to Gaussian and show the image.
>>> img_fit = img_med.gaussfit(scale=1/8)
>>> img_fit.imshow()
Normalize the image and save it.
>>> img *= (img_fit.max() / img_fit)
>>> img.imsave()

(ex.2) Drift correction of multi-channel image
>>> img2 = img.drift_correction(ref=img["c=1"])

(ex.3) Deconvolution
>>> psfinfo = {"wavelength":0.57}
>>> img2.set_scale(x=0.15, y=0.15, z=0.22)
>>> img2 = img.lucy(psf=psfinfo, niter=20)


Inheritance
-----------
              ____    MetaArray   _______
             /              \            \
       HistoryArray        MarkerArray  PropArray
       /         \          /
 LabeledArray   Label  IndexArray
    /
ImgArray

"""

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)

