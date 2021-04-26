import numpy as np
from scipy import optimize as opt
from scipy.stats import entropy
from tifffile import TiffFile
from skimage.morphology import disk, ball
from skimage import transform as sktrans
import json
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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

def check_nd_sigma(sigma, ndim):
    if np.isscalar(sigma):
        sigma = [sigma] * ndim
    elif len(sigma) != ndim:
        raise ValueError("length of sigma and dims must match.")
    return sigma


def specify_one(center, radius, shape:tuple, labeltype:str):
    if labeltype == "square":
        sl = (...,) + tuple(slice(xc-r, xc+r+1, None) for xc, r in zip(center, radius))
    elif labeltype == "ellipse":
        ind = np.indices(shape)
        # (x-x_0)^2/r_x^2 + (y-y_0)^2/r_y^2 + (z-z_0)^2/r_z^2 <= 1
        sl = sum([((i-xc)/r)**2 for i, xc, r in zip(ind, center, radius)]) <= 1.0
    elif labeltype == "circle":
        r = radius[0]
        if not (radius == r).all():
            raise ValueError("Cannot set different radii when shape is 'circle'")
        
        sl = np.zeros(shape, dtype=bool)
        area = ball_like(r, len(center)).astype(bool)
        r_ = area.shape[0]//2
        bbox = tuple(slice(xc - r_, xc + r_ + 1, None) for xc in center)
        try:
            sl[bbox] = area
        except ValueError:
            pass
        
    else:
        raise ValueError(f"{shape}")
    
    return sl

def check_matrix(ref):
    """
    Check Affine transformation matrix
    """    
    mtx = []
    for m in ref:
        if np.isscalar(m): 
            if m == 1:
                mtx.append(m)
            else:
                raise ValueError(f"Only `1` is ok, but got {m}")
            
        elif m.shape != (3, 3) or not np.allclose(m[2,:2], 0):
            raise ValueError(f"Wrong Affine transformation matrix:\n{m}")
        
        else:
            mtx.append(m)
    return mtx

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

def plot_drift(result):
    fig = plt.figure()
    ax = fig.add_subplot(111, title="drift")
    if result.shape[0] == 2:
        ax.plot(result[1], result[0], marker="+", color="red")
    elif result.shape[1] == 2:
        ax.plot(result[:, 1], result[:, 0], marker="+", color="red")
    else:
        raise ValueError(f"Wrong shape {result.shape}")
    ax.grid()
    # delete the default axes and let x=0 and y=0 be new ones.
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # let the interval of x-axis and that of y-axis be equal.
    ax.set_aspect("equal")
    # set the x/y-tick intervals to 1.
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    return None

def plot_gaussfit_result(raw, fit):
    x0 = raw.shape[1]//2
    y0 = raw.shape[0]//2
    plt.figure(figsize=(6,4))
    plt.subplot(2, 1, 1)
    plt.title("x-direction")
    plt.plot(raw[y0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[y0], color="red", label="fit")
    plt.subplot(2, 1, 2)
    plt.title("y-direction")
    plt.plot(raw[:,x0].value, color="gray", alpha=0.5, label="raw image")
    plt.plot(fit[:,x0], color="red", label="fit")
    plt.tight_layout()
    plt.show()
    return None
    

def key_repr(key):
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
        elif s is ...:
            keylist.append("...")
        else:
            keylist.append(str(s))
    
    return ",".join(keylist)

def circle(radius, shape, dtype="bool"):
    x = np.arange(-(shape[0] - 1) / 2, (shape[0] - 1) / 2 + 1)
    y = np.arange(-(shape[1] - 1) / 2, (shape[1] - 1) / 2 + 1)
    dx, dy = np.meshgrid(x, y)
    return np.array((dx ** 2 + dy ** 2) <= radius ** 2, dtype=dtype)

def ball_like(radius, ndim:int):
    if ndim == 1:
        return np.ones(int(radius)*2+1, dtype=np.uint8)
    elif ndim == 2:
        return disk(radius)
    elif ndim == 3:
        return ball(radius)
    else:
        raise ValueError(f"dims must be 1 - 3, but got {ndim}")

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

def add_axes(axes, shape, arr2d):
    """
    stack yx-ordered array 'arr2d' to 'axes' in shape 'shape'
    """
    if len(shape) == 2:
        return arr2d
    arr2d = np.array(arr2d)
    for i, o in enumerate(reversed(axes)):
        if o not in "yx":
            arr2d = np.stack([arr2d]*(shape[-i-1]))
    return arr2d



def determine_range(arr):
    """
    Called in imshow()
    """
    if arr.dtype == bool:
        vmax = vmin = None
    else:
        try:
            vmax = np.percentile(arr[arr>0], 99.99)
            vmin = np.percentile(arr[arr>0], 0.01)
        except IndexError:
            vmax = vmin = None
    return vmax, vmin

def determine_dims(img):
    dims = len(img.spatial_shape)
    if dims not in (2, 3):
        raise ValueError("Image must be 2 or 3 dimensional.")
    return dims

def determine_spatial_dims(dims:int):
    if dims == 2:
        dims = "yx"
    elif dims == 3:
        dims = "zyx"
    else:
        raise ValueError(f"dimension must be 2 or 3, but got {dims}")
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
    psf = np.asarray(psf, dtype="float32")
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