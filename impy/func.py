import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import optimize as opt
from scipy.stats import entropy
from tifffile import TiffFile
from skimage.morphology import disk, ball
from skimage import transform as sktrans
from functools import wraps
import time
import json
import re

class Timer:
    def __init__(self):
        self.tic()
        
    def tic(self):
        self.t = time.time()
    
    def toc(self):
        self.t = time.time() - self.t
    
    def __str__(self):
        minute, sec = divmod(self.t, 60)
        sec = np.round(sec, 2)
        if minute == 0:
            out = f"{sec} sec"
        else:
            out = f"{int(minute)} min {sec} sec"
        return out

def load_json(s:str):
    return json.loads(re.sub("'", '"', s))
    
def get_meta(path:str):
    with TiffFile(path) as tif:
        hist = []
        ijmeta = tif.imagej_metadata
        if ijmeta is None:
            ijmeta = {}
        
        ijmeta.pop("ROI", None)
        
        if "Info" in ijmeta.keys():
            try:
                infodict = load_json(ijmeta["Info"])
            except:
                infodict = {}
            if "impyhist" in infodict.keys():
                hist = infodict["impyhist"].split("->")
        
        try:
            axes = tif.series[0].axes.lower()
        except:
            axes = None
    
    
    return {"axes":axes, "ijmeta":ijmeta, "history":hist}

def record(func):
    """
    Record the name of ongoing function.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.ongoing = func.__name__
        out = func(self, *args, **kwargs)
        self.ongoing = None
        del self.ongoing
        return out
    return wrapper

def same_dtype(asfloat=False):
    """
    Decorator to assure output image has the same dtype as the input
    image. 

    Parameters
    ----------
    asfloat : bool, optional
        If input image should be converted to float first, by default False
    """    
    def _same_dtype(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            dtype = self.dtype
            if asfloat:
                self = self.astype("float32")
            out = func(self, *args, **kwargs)
            out = out.as_img_type(dtype)
            return out
        return wrapper
    return _same_dtype

def gauss2d(x, y, mu1, mu2, sg1, sg2, A, B):
    """
    e.g. for 100x200 image,
    x = [0, 1, 2, 3, ..., 100] 
    y = [0, 1, 2, 3, ..., 200]
    """
    x, y = np.meshgrid(y, x)
    
    z = A*np.exp(-((((x - mu1)/sg1)**2) + ((y - mu2)/sg2)**2)/2) + B
    
    return z


def square(params, func, z):
    """
    calculate ||z - func(x, y, *params)||^2
    where x and y are determine by z.shape
    """
    x = np.arange(z.shape[0])
    y = np.arange(z.shape[1])
    z_guess = func(x, y, *params)
    return np.mean((z - z_guess)**2)
    
def gaussfit(img2d, p0=None, scale=1, show_result=True):
    """
    Fit 2-D image to 2-D gaussian

    Parameters
    ----------
    img2d : 2-D image
    p0 : initial parameters
    scale : float, optional
    """
    
    rough = img2d.rescale(scale).value.astype("float32")
    
    if p0 is None:
        mu1, mu2 = np.unravel_index(np.argmax(rough), rough.shape)  # 2-dim argmax
        sg1 = rough.shape[0]
        sg2 = rough.shape[1]
        B = np.percentile(rough, 5)
        A = np.percentile(rough, 95) - B
        p0 = mu1, mu2, sg1, sg2, A, B
    
    param = opt.minimize(square, p0, args=(gauss2d, rough)).x
    param[:4] /= scale
    
    x = np.arange(img2d.shape[0])
    y = np.arange(img2d.shape[1])

    fit = gauss2d(x, y, *param).astype("float32")
    
    # show fitting result
    if show_result:
        x0 = img2d.shape[1]//2
        y0 = img2d.shape[0]//2
        plt.figure(figsize=(6,4))
        plt.subplot(2,1,1)
        plt.title("x-direction")
        plt.plot(img2d[y0].value, color="gray", alpha=0.5, label="raw image")
        plt.plot(fit[y0], color="red", label="fit")
        plt.subplot(2,1,2)
        plt.title("y-direction")
        plt.plot(img2d[:,x0].value, color="gray", alpha=0.5, label="raw image")
        plt.plot(fit[:,x0], color="red", label="fit")
        plt.tight_layout()
        plt.show()
    return param, fit


def affinefit(img, imgref, bins=256, order=3):
    as_3x3_matrix = lambda mtx: np.vstack((mtx.reshape(2,3), [0., 0., 1.]))
    
    def normalized_mutual_information(img, imgref):
        """
        Y(A,B) = (H(A)+H(B))/H(A,B)
        See "Elegant SciPy"
        """
        hist, edges = np.histogramdd([np.ravel(img), np.ravel(imgref)], bins=bins)
        hist /= np.sum(hist)
        e1 = entropy(np.sum(hist, axis=0)) # Shannon entropy
        e2 = entropy(np.sum(hist, axis=1))
        e12 = entropy(np.ravel(hist)) # mutual entropy
        return (e1 + e2)/e12
    
    def cost_nmi(mtx, img, imgref):
        mtx = sktrans.AffineTransform(matrix=as_3x3_matrix(mtx))
        img_transformed = sktrans.warp(img, mtx, order=order)
        return -normalized_mutual_information(img_transformed, imgref)
    
    mtx0 = np.array([[1., 0., 0.],
                     [0., 1., 0.]]) # aberration makes little difference
    
    result = opt.minimize(cost_nmi, mtx0, args=(np.asarray(img), np.asarray(imgref)), method="Powell")
    mtx_opt = as_3x3_matrix(result.x)
    return mtx_opt
    

def _key_repr(key):
    keylist = []
        
    if isinstance(key, tuple):
        _keys = key
    elif hasattr(key, "__array__"):
        _keys = ("array",)
    else:
        _keys = (key,)
    
    for s in _keys:
        if isinstance(s, (slice, list, np.ndarray)):
            keylist.append("*")
        elif s is None:
            keylist.append("new")
        else:
            keylist.append(str(s))
    
    return ",".join(keylist)

def circle(radius, shape, dtype="bool"):
    x = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    dx, dy = np.meshgrid(x, y)
    return np.array((dx ** 2 + dy ** 2) <= radius ** 2, dtype=dtype)

def ball_like(radius, dims:int):
    if dims == 2:
        return disk(radius)
    elif dims == 3:
        return ball(radius)
    else:
        raise ValueError(f"dims must be 2 or 3, but got {dims}")

def del_axis(axes, axis):
    """
    axes: str or Axes object.
    axis: int.
    delete axis from axes.
    """
    if type(axis) == int:
        axis = [axis]
    if axes is None:
        return None
    new_axes = ""
    for i, o in enumerate(axes):
        if (i not in axis):
            new_axes += o
            
    return new_axes

def add_axes(axes, shape, arr2d):
    """
    stack yx-ordered array 'arr2d' to 'axes' in shape 'shape'
    """
    if len(shape) == 2:
        return arr2d
    arr2d = np.array(arr2d)
    for i, o in enumerate(reversed(axes)):
        if (o not in "yx"):
            arr2d = np.stack([arr2d]*(shape[-i-1]))
    return arr2d


def get_lut(name):
    try:
        lut = plt.get_cmap(name)
    except:
        try:
            lut = LinearSegmentedColormap.from_list(name + "_cmap", ["black", name])
        except:
            print(f"{name} is not a color or a cmap.")
            lut = "gray"
    return lut

def determine_range(arr):
    if arr.dtype == bool:
        vmax = vmin = None
    else:
        vmax = np.percentile(arr[arr>0], 99.99)
        vmin = np.percentile(arr[arr>0], 0.01)
    return vmax, vmin