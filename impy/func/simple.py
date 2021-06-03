from __future__ import annotations
import numpy as np
from tifffile import TiffFile
import json
import re
from scipy.spatial.transform import Rotation

def load_json(s:str):
    return json.loads(re.sub("'", '"', s))
    
def get_meta(path:str):
    with TiffFile(path) as tif:
        ijmeta = tif.imagej_metadata
        series0 = tif.series[0]
    
    pagetag = series0.pages[0].tags
    
    hist = []
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
        axes = series0.axes.lower()
    except:
        axes = None
    
    tags = {v.name: v.value for v in pagetag.values()}
    
    return {"axes": axes, "ijmeta": ijmeta, "history": hist, "tags": tags}


def check_nd(x, ndim:int):
    if np.isscalar(x):
        x = (x,) * ndim
    elif len(x) != ndim:
        raise ValueError("length of parameter and dimension must match.")
    return x


def specify_one(center, radius, shape:tuple) -> tuple[slice]:
    sl = tuple(slice(max(0, xc-r), min(xc+r+1, sh), None) 
                        for xc, r, sh in zip(center, radius, shape))
    return sl


def gabor_kernel_nd(lmd, theta, psi:float, sigma:float, gamma:float, radius:int, ndim:int):
    if ndim == 2:
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    elif ndim == 3:
        rot = Rotation.from_rotvec(theta).as_matrix()
    else:
        raise NotImplementedError
    sl = slice(-radius, radius+1)
    r = np.stack(np.mgrid[(sl,)*ndim])
    # r'_izyx = R_id * r_dzyx
    r_rot = np.tensordot(rot, r, ([1], [0]))
    ker = np.empty(r_rot[0].shape, dtype=np.complex64)
    chi = r_rot[0]**2 + gamma*np.sum(r_rot[1:]**(ndim-1))
    ker[:] = np.exp(chi/(2*sigma**2)) \
        / 2 * np.pi * (sigma**ndim) * (gamma**(ndim-1)) \
        * np.exp(1j * (2*np.pi*r_rot[0]/lmd + psi))
        
    return ker
    
def find_first_appeared(axes, include="", exclude=""):
    for a in axes:
        if a in include or not a in exclude:
            return a
    raise ValueError(f"Inappropriate axes: {axes}")

def del_axis(axes, axis) -> str:
    """
    axes: str or Axes object.
    axis: int.
    delete axis from axes.
    """
    new_axes = ""
    if isinstance(axis, int):
        axis = [axis]
    if axes is None:
        return None
    else:
        axes = str(axes)
    
    if isinstance(axis, list):
        for i, o in enumerate(axes):
            if i not in axis:
                new_axes += o
    elif isinstance(axis, str):
        new_axes = complement_axes(axis, axes)
        
    return new_axes

def add_axes(axes, shape, key, key_axes="yx"):
    """
    Stack `key` to make its shape key_axes-> axes.
    """    
    if shape == key.shape:
        return key
    key = np.asarray(key)
    for i, o in enumerate(axes):
        if o not in key_axes:
            key = np.stack([key]*(shape[i]), axis=i)
    return key

def determine_range(arr):
    """
    Called in imshow()
    """
    if arr.dtype == bool:
        vmax = 1
        vmin = 0
    else:
        try:
            vmax = np.percentile(arr[arr>0], 99.99)
            vmin = np.percentile(arr[arr>0], 0.01)
        except IndexError:
            vmax = arr.max()
            vmin = arr.min()
    return vmax, vmin

def determine_dims(img):
    dims = len(img.spatial_shape)
    if dims not in (2, 3):
        raise ValueError("Image must be 2 or 3 dimensional.")
    return dims

def check_clip_range(in_range, img):
    """
    Called in clip_outliers() and rescale_intensity().
    """    
    lower, upper = in_range
    if isinstance(lower, str) and lower.endswith("%"):
        lower = float(lower[:-1])
        lowerlim = np.percentile(img, lower)
    elif lower is None:
        lowerlim = np.min(img)
    else:
        lowerlim = float(lower)
    
    if isinstance(upper, str) and upper.endswith("%"):
        upper = float(upper[:-1])
        upperlim = np.percentile(img, upper)
    elif upper is None:
        upperlim = np.max(img)
    else:
        upperlim = float(upper)
    
    if lowerlim >= upperlim:
        raise ValueError(f"lowerlim is larger than upperlim: {lowerlim} >= {upperlim}")
    
    return lowerlim, upperlim

def axes_included(img, label):
    """
    e.g.)
    img.axes = "tyx", label.axes = "yx" -> True
    img.axes = "tcyx", label.axes = "zyx" -> False
    
    """
    return all([a in img.axes for a in label.axes])

def shape_match(img, label):
    """
    e.g.)
    img   ... 12(t), 100(y), 50(x)
    label ... 100(y), 50(x)
        -> True
    img   ... 12(t), 100(y), 50(x)
    label ... 30(y), 50(x)
        -> False
    """    
    return all([img.sizeof(a)==label.sizeof(a) for a in label.axes])

def complement_axes(axes, all_axes="ptzcyx"):
    c_axes = ""
    for a in all_axes:
        if a not in axes:
            c_axes += a
    return c_axes

def check_psf(img, psf, dims):
    psf = np.asarray(psf, dtype=np.float32)
    psf /= np.max(psf)
    
    if img.sizesof(dims) != psf.shape:
        raise ValueError("observation and PSF have different shape: "
                        f"{img.sizesof(dims)} and {psf.shape}")
    return psf

def check_filter_func(f):
    if f is None:
        f = lambda x: True
    elif not callable(f):
        raise TypeError("`filt` must be callable.")
    return f


def largest_zeros(shape) -> np.ndarray:
    try:
        out = np.zeros(shape, dtype=np.uint64)
    except MemoryError:
        try:
            out = np.zeros(shape, dtype=np.uint32)
        except MemoryError:
            out = np.zeros(shape, dtype=np.uint16)
    return out

def radial_profile(data, center, scales):
    ind = np.indices((data.shape))
    r = np.sqrt(sum(((x - c)/s)**2 for x, c, s in zip(ind, center, scales)))
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

def iter_radial_profile(data, center, scales):
    ind = np.indices((data.shape))
    r = np.sqrt(sum(((x - c)/s)**2 for x, c, s in zip(ind, center, scales)))
    ...