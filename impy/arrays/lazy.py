from __future__ import annotations
from ..func import *
from dask import array as da
from ..axes import Axes, ImageAxesError
from .imgarray import ImgArray
from scipy import ndimage as ndi
import itertools
from .labeledarray import _make_rotated_axis
# TODO: crop_center etc, binning?

class LazyImgArray:
    MAX_GB = 2.0
    additional_props = ["dirpath", "metadata", "name"]
    def __init__(self, obj: da.core.Array, name=None, axes=None, dirpath=None, history=None, metadata=None):
        if not isinstance(obj, (da.core.Array,)):
            raise TypeError(f"obj must be dask array, got {type(obj)}")
        self.img = obj
        self.dirpath = dirpath
        self.name = name
        
        # MicroManager
        if isinstance(self.name, str) and self.name.endswith("_MMStack_Pos0.ome"):
            self.name = self.name[:-17]
        
        self.axes = axes
        self.metadata = metadata
        self.history = [] if history is None else history
        
    @property
    def ndim(self):
        return self.img.ndim
    
    @property
    def shape(self):
        return self.img.shape
    
    @property
    def dtype(self):
        return self.img.dtype
    
    @property
    def size(self):
        return self.img.size
    
    @property
    def axes(self):
        return self._axes
    
    @property
    def itemsize(self):
        return self.img.itemsize
    
    @property
    def chunksize(self):
        return self.img.chunksize
    
    @property
    def gb(self):
        return self.size * self.itemsize / 1e9
    
    @axes.setter
    def axes(self, value):
        if value is None:
            self._axes = Axes()
        else:
            self._axes = Axes(value)
            if self.ndim != len(self._axes):
                raise ImageAxesError("Inconpatible dimensions: "
                                    f"image (ndim={self.ndim}) and axes ({value})")
    
    def __array__(self):
        return np.asarray(self.img)
    
    def __getitem__(self, key):
        if isinstance(key, str):
            key = axis_targeted_slicing(self.img, self.axes, key)
        keystr = key_repr(key) # write down key like "0,*,*"
        
        if hasattr(key, "__array__"):
            # fancy indexing will lose axes information
            new_axes = None
            
        elif "new" in keystr:
            # np.newaxis or None will add dimension
            new_axes = None
            
        elif self.axes:
            del_list = [i for i, s in enumerate(keystr.split(",")) if s not in ("*", "")]
            new_axes = del_axis(self.axes, del_list)
        else:
            new_axes = None
        out = self.__class__(self.img[key], name=self.name, dirpath=self.dirpath, axes=new_axes, 
                             metadata=self.metadata, history=self.history)
        
        out._getitem_additional_set_info(self, keystr=keystr,
                                         new_axes=new_axes, key=key)
        
        return out
    
    @property
    def shape_info(self):
        if self.axes.is_none():
            shape_info = self.shape
        else:
            shape_info = ", ".join([f"{s}({o})" for s, o in zip(self.shape, self.axes)])
        return shape_info
    
    @property
    def scale(self):
        return self.axes.scale
    
    @property
    def scale_unit(self):
        try:
            unit = self.metadata["unit"]
            if unit.startswith(r"\u"):
                unit = "u" + unit[6:]
        except Exception:
            unit = None
        return unit
    
    @scale_unit.setter
    def scale_unit(self, unit):
        if not isinstance(unit, str):
            raise TypeError("Can only set str to scale unit.")
        if isinstance(self.metadata, dict):
            self.metadata["unit"] = unit
        else:
            self.metadata = {"unit": unit}
    
    def set_scale(self, other=None, **kwargs) -> None:
        """
        Set scales of each axis.

        Parameters
        ----------
        other : dict or MetaArray, optional
            New scales. If dict, it should be like {"x": 0.1, "y": 0.1}. If MetaArray, only
            scales of common axes are copied.
        kwargs : 
            This enables function call like set_scale(x=0.1, y=0.1).

        """        
        if self.axes.is_none():
            raise ImageAxesError("Image does not have axes.")
        
        elif isinstance(other, dict):
            # lateral-scale can be set with one keyword.
            if "yx" in other:
                yxscale = other.pop("yx")
                other["x"] = other["y"] = yxscale
            if "xy" in other:
                yxscale = other.pop("xy")
                other["x"] = other["y"] = yxscale
            # check if all the keys are contained in axes.
            for a, val in other.items():
                if a not in self.axes:
                    raise ImageAxesError(f"Image does not have axis {a}.")    
                elif not np.isscalar(val):
                    raise TypeError(f"Cannot set non-numeric value as scales.")
            self.axes.scale.update(other)
            
        elif kwargs:
            self.set_scale(dict(kwargs))
        
        elif isinstance(other, self.__class__):
            self.set_scale({a: s for a, s in other.scale.items() if a in self.axes})
        
        else:
            raise TypeError(f"'other' must be str or LazyImgArray, but got {type(other)}")
        
        return None
    
    
    def _repr_dict_(self):
        return {"    shape     ": self.shape_info,
                "    dtype     ": self.dtype,
                "  directory   ": self.dirpath,
                "original image": self.name}
    
    def __repr__(self):
        return "\n" + "\n".join(f"{k}: {v}" for k, v in self._repr_dict_().items()) + "\n"
    
    @property
    def data(self) -> ImgArray:
        if self.gb > self.__class__.MAX_GB:
            raise RuntimeError(f"Too large: {self.gb:.2f} GB")
        
        img = self.img.compute().view(ImgArray)
        for attr in ["name", "dirpath", "axes", "metadata", "history"]:
            setattr(img, attr, getattr(self, attr, None))
        return img
    
    
    def rotated_crop(self, origin, dst1, dst2) -> LazyImgArray:
        """
        Crop the image at four courners of an rotated rectangle. Currently only supports rotation within 
        yx-plane. An rotated rectangle is specified with positions of a origin and two destinations `dst1`
        and `dst2`, i.e., vectors (dst1-origin) and (dst2-origin) represent a rotated rectangle. Let 
        origin be the origin of a xy-plane, the rotation direction from dst1 to dst2 must be counter-
        clockwise, or the cropped image will be reversed.
        
        Parameters
        ---------- 
        origin : (float, float)
        dst1 : (float, float)
        dst2 : (float, float)
        """
        origin = np.asarray(origin)
        dst1 = np.asarray(dst1)
        dst2 = np.asarray(dst2)
        ax0 = _make_rotated_axis(origin, dst2)
        ax1 = _make_rotated_axis(dst1, origin)
        all_coords = ax0[:, np.newaxis] + ax1[np.newaxis] - origin
        all_coords = np.moveaxis(all_coords, -1, 0)
        
        chunks = self.chunksize[:-2] + all_coords.shape[-2:]
        cropped_img = da.empty(self.shape[:-2] + all_coords.shape[1:], dtype=self.dtype, chunks=chunks)
        iters = itertools.product(*map(range, self.shape[:-2]))
        for sl in iters:
            cropped_img[sl] = self.img[sl].map_blocks(ndi.map_coordinates, coordinates=all_coords, 
                                                      prefilter=False, order=1, drop_axis=[0,1], 
                                                      dtype=self.dtype)
        out = self.__class__(cropped_img)
        out._set_info(self, f"rotated_crop")
        return out
    
    def axisof(self, axisname):
        if type(axisname) is int:
            return axisname
        else:
            return self.axes.find(axisname)
    
    def sizeof(self, axis:str):
        return self.shape[self.axes.find(axis)]
    
    def sizesof(self, axes:str):
        return tuple(self.sizeof(a) for a in axes)
    
    def transpose(self, axes):
        if self.axes.is_none():
            new_axes = None
        else:
            new_axes = "".join([self.axes[i] for i in list(axes)])
        out = self.__class__(self.img.transpose(axes))
        out._set_info(self, new_axes=new_axes)
        return out
    
    def crop_center(self, scale=0.5, *, dims="yx") -> LazyImgArray:
        """
        Crop out the center of an image. 
        
        Parameters
        ----------
        scale : float or array-like, default is 0.5
            Scale of the cropped image. If an array is given, each axis will be cropped in different scales,
            using each value respectively.
        dims : str, default is "yx"
            Dimensions to be cropped.
            
        Example
        -------
        (1) Create 512x512 image from 1024x1024 image.
        >>> img_cropped = img.crop_center(scale=0.5)
        (2) Create 21x256x256 image from 63x1024x1024 image.
        >>> img_cropped = img.crop_center(scale=[1/3, 1/2, 1/2])
        """
        # check scale
        if hasattr(scale, "__iter__") and len(scale) == 3 and dims == "yx":
            dims = "zyx"
        scale = np.asarray(check_nd(scale, len(dims)))
        if np.any((scale <= 0) | (1 < scale)):
            raise ValueError(f"scale must be (0, 1], but got {scale}")
        
        # Make axis-targeted slicing string
        sizes = self.sizesof(dims)
        slices = []
        for a, size, sc in zip(dims, sizes, scale):
            x0 = int(size / 2 * (1 - sc))
            x1 = int(np.ceil(size / 2 * (1 + sc)))
            slices.append(f"{a}={x0}:{x1}")

        out = self[";".join(slices)]
        
        return out
    
    def proj(self, axis:str=None, method:str="mean") -> LazyImgArray:
        """
        Z-projection along any axis.

        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        method : str , default is mean-projection.
            Projection method. If str is given, it will converted to numpy function.

        Returns
        -------
        LazyImgArray
            Projected image.
        """        
        if axis is None:
            axis = find_first_appeared(self.axes, include=self.axes, exclude="yx")
        elif not isinstance(axis, str):
            raise TypeError("`axis` must be str.")
        axisint = [self.axisof(a) for a in axis]
        if method == "mean":
            projection = getattr(self.img, method)(axis=tuple(axisint), dtype=self.dtype)
        else:
            projection = getattr(self.img, method)(axis=tuple(axisint))
        out = self.__class__(projection)
        out._set_info(self, f"proj(axis={axis}, method={method})", del_axis(self.axes, axisint))
        return out
    
    # TODO: how to lazily convert image type?
    # def as_uint8(self) -> LazyImgArray:
    #     img = self.img
    #     if img.dtype == np.uint8:
    #         return img
        
    #     if img.dtype == np.uint16:
    #         out = img / 256
    #     elif img.dtype.kind == "f":
    #         out = lazy_clip_float(img, 256)
    #     else:
    #         raise TypeError(f"invalid data type: {img.dtype}")
    #     out = out.astype(np.uint8)
    #     out = self.__class__(out)
    #     out._set_info(self)
    #     return out
    
    # def as_uint16(self) -> LazyImgArray:
    #     img = self.img
    #     if img.dtype == np.uint16:
    #         return img
    #     if img.dtype == np.uint8:
    #         out = img * 256
    #     elif img.dtype == bool:
    #         out = img
    #     elif img.dtype.kind == "f":
    #         out = lazy_clip_float(img, 256)
    #     else:
    #         raise TypeError(f"invalid data type: {img.dtype}")
    #     out = out.astype(np.uint16)
    #     out = self.__class__(out)
    #     out._set_info(self)
    #     return out
    
    def as_float(self) -> LazyImgArray:
        out = self.img.astype(np.float32)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def as_img_type(self, dtype=np.uint16) -> LazyImgArray:
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype == "float32":
            return self.as_float()
        else:
            raise ValueError(f"dtype: {dtype}")
    
    def _set_additional_props(self, other):
        # set additional properties
        # If `other` does not have it and `self` has, then the property will be inherited.
        for p in self.__class__.additional_props:
            setattr(self, p, getattr(other, p, 
                                     getattr(self, p, 
                                             None)))
    
    
    def _getitem_additional_set_info(self, other, **kwargs):
        keystr = kwargs["keystr"]
        self._set_info(other, f"getitem[{keystr}]", kwargs["new_axes"])
        return None
    
    def _set_info(self, other, next_history=None, new_axes:str="inherit"):
        self._set_additional_props(other)
        # set axes
        try:
            if new_axes != "inherit":
                self.axes = new_axes
                self.set_scale(other)
            else:
                self.axes = other.axes.copy()
        except ImageAxesError:
            self.axes = None
        
        # set history
        if next_history is not None:
            self.history = other.history + [next_history]
        else:
            self.history = other.history.copy()
        
        return None
    
# @dask.delayed
# def lazy_clip_float(img, upper):
#     if 0 <= img.min() and img.max() < 1:
#         out = img * upper
#     else:
#         out = img + 0.5
#     out = da.clip(out, 0, upper-1)
#     return out