__version__ = "1.0.0"
import warnings
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)
warnings.simplefilter("ignore", ResourceWarning)
warnings.simplefilter("ignore", RuntimeWarning)

import os
from importlib import import_module
from .imgarray import array, imread, imread_collection, read_meta, stack, set_cpu, ImgArray

__doc__ = \
r"""
Extended version of NumPy for image analysis.
The name of original image and executed calculations are all recorded.
(See also __array_finalize__, __array_ufunc__ and the NumPy official documents)

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
>>> img_fit = img_med.rough_gaussfit(scale=1/8)
>>> img_fit.imshow()
Normalize the image and save it.
>>> img *= (img_fit.max() / img_fit)
>>> img.imsave()

(ex.2) Drift correction of multi-channel image
>>> img2 = img.drift_correction(ref=img["c=1"])

(ex.3) Deconvolution
>>> psfinfo = {"wavelength":0.57, "pxsize":0.1, "dz":0.3}
>>> img2 = img.lucy3d(psfinfo=psfinfo, niter=20)

"""

# load plugins if possible
plugin_path = os.path.join(os.path.dirname(__file__), "plugin")
py_file_list = []
for file in os.listdir(plugin_path):
    name, ext = os.path.splitext(file)
    if (ext == ".py" and not name.startswith("__")):
        py_file_list.append(name)

plugin_func_list = []   # list of plugin function objects
for py in py_file_list:
    try:
        mod = import_module(f".plugin.{py}", "impy")
    except ImportError as e:
        print(f"Could not load '{py}': {e}")
    else:
        for func in mod.__all__:
            plugin_func_list.append(getattr(mod, func))
            
plugin = []             # list of plugin function names
for func in plugin_func_list:
    setattr(ImgArray, func.__name__, func)
    plugin.append(func.__name__)

del os, plugin_func_list, plugin_path, import_module

# To silence Warnings in skimage
warnings.resetwarnings()
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", DeprecationWarning)



