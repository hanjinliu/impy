import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tifffile import TiffFile
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
        minute, sec = divmod(int(self.t), 60)
        if (minute == 0):
            out = f"{sec} sec"
        else:
            out = f"{minute} min {sec} sec"
        return out

def load_json(s:str):
    return json.loads(re.sub("'", '"', s))
    
def get_meta(path:str):
    with TiffFile(path) as tif:
        hist = []
        ijmeta = tif.imagej_metadata
        if (ijmeta is None):
            ijmeta = {}
        
        ijmeta.pop("ROI", None)
        
        if ("Info" in ijmeta.keys()):
            try:
                infodict = load_json(ijmeta["Info"])
            except:
                infodict = {}
            if ("impyhist" in infodict.keys()):
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
            if (asfloat):
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
    
    z = A*np.exp(-((((x - mu1)/sg1)**2) + (((y - mu2)/sg2)**2))/2) + B
    
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

def circle(radius, shape, dtype="bool"):
    x = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    dx, dy = np.meshgrid(x, y)
    return np.array((dx ** 2 + dy ** 2) <= radius ** 2, dtype=dtype)


def del_axis(axes, axis):
    """
    axes: str or Axes object.
    axis: int.
    delete axis from axes.
    """
    if (type(axis) == int):
        axis = [axis]
    if (axes is None):
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
    if (len(shape) == 2):
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
