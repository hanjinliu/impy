from __future__ import annotations
from dask import array as da
from scipy import ndimage as ndi
import itertools
from ..deco import *
from ..func import *
from ..axes import ImageAxesError
from .imgarray import ImgArray
from .labeledarray import _make_rotated_axis
from .axesmixin import AxesMixin

class LazyImgArray(AxesMixin):
    MAX_GB = 2.0
    additional_props = ["dirpath", "metadata", "name"]
    def __init__(self, obj: da.core.Array, name=None, axes=None, dirpath=None, history=None, metadata=None):
        if not isinstance(obj, da.core.Array):
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
    def itemsize(self):
        return self.img.itemsize
    
    @property
    def chunksize(self):
        return self.img.chunksize
    
    @property
    def gb(self):
        return self.size * self.itemsize / 1e9
    
    def __array__(self):
        return self.img.compute()
    
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
    def chunk_info(self):
        if self.axes.is_none():
            chunk_info = self.chunksize
        else:
            chunk_info = ", ".join([f"{s}({o})" for s, o in zip(self.chunksize, self.axes)])
        return chunk_info
    
    def _repr_dict_(self):
        return {"    shape     ": self.shape_info,
                " chunk sizes  ": self.chunk_info,
                "    dtype     ": self.dtype,
                "  directory   ": self.dirpath,
                "original image": self.name,
                "   history    ": "->".join(self.history)}
    
    def __repr__(self):
        return "\n" + "\n".join(f"{k}: {v}" for k, v in self._repr_dict_().items()) + "\n"
    
    @property
    def data(self) -> ImgArray:
        if self.gb > self.__class__.MAX_GB:
            raise RuntimeError(f"Too large: {self.gb:.2f} GB")
        with Progress("Computing Dask"):
            img = self.img.compute().view(ImgArray)
            for attr in ["name", "dirpath", "axes", "metadata", "history"]:
                setattr(img, attr, getattr(self, attr, None))
        return img
    
    def apply(self, funcname:str, *args, **kwargs) -> LazyImgArray:
        """
        Apply dask array function to the connected dask array.

        Parameters
        ----------
        funcname : str
            Name of function to apply.
        args, kwargs :
            Parameters that will be passed to `funcname`.

        Returns
        -------
        LazyImgArray
            Updated one
        """        
        
        out = getattr(self.img, funcname)(*args, **kwargs)
        out = self.__class__(out)
        new_axes = "inherit" if out.shape == self.shape else None
        out._set_info(self, make_history(funcname, args, kwargs), new_axes=new_axes)
        return out
    
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
        
        # rechunk to assign xy-plane to the same chunk.
        if self.shape[-2:] != self.chunksize[-2:]:
            new_chunks = ("auto",)*(self.ndim-2) + self.shape[-2:]
            input_img = self.img.rechunk(chunks=new_chunks)
        else:
            input_img = self.img
        
        chunks = self.chunksize[:-2] + all_coords.shape[-2:]
        cropped_img = da.empty(self.shape[:-2] + all_coords.shape[1:], dtype=self.dtype, chunks=chunks)
        iters = itertools.product(*map(range, self.shape[:-2]))
        for sl in iters:
            cropped_img[sl] = input_img[sl].map_blocks(ndi.map_coordinates, coordinates=all_coords, 
                                                      prefilter=False, order=1, drop_axis=[0,1], 
                                                      dtype=self.dtype)
        out = self.__class__(cropped_img)
        out._set_info(self, "rotated_crop")
        return out
    
    
    def chunksizeof(self, axis:str):
        return self.img.chunksize[self.axes.find(axis)]
    
    def chunksizesof(self, axes:str):
        return tuple(self.chunksizeof(a) for a in axes)
    
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
    
    @same_dtype()
    def proj(self, axis:str=None, method:str="mean", chunks=None) -> LazyImgArray:
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
            axis = find_first_appeared("ztpi<c", include=self.axes, exclude="yx")
        elif not isinstance(axis, str):
            raise TypeError("`axis` must be str.")
        axisint = [self.axisof(a) for a in axis]
        
        # rechunk if array is split into too many chunks along `axis`
        if chunks is None:
            chunks = []
            for i in range(self.ndim):
                if i in axisint:
                    chunks.append(self.shape[i] % 2000)
                else:
                    chunks.append("auto")
        if any(c != "auto" for c in chunks):
            input_img = self.img.rechunk(chunks=chunks)
        
        if method == "mean":
            projection = getattr(input_img, method)(axis=tuple(axisint), dtype=self.dtype)
        else:
            projection = getattr(input_img, method)(axis=tuple(axisint))
        out = self.__class__(projection)
        out._set_info(self, f"proj(axis={axis}, method={method})", del_axis(self.axes, axisint))
        return out
    
    @dims_to_spatial_axes
    @same_dtype()
    def binning(self, binsize:int=2, method="sum", *, check_edges=True, chunks=None, dims=None) -> LazyImgArray:
        """
        Binning of images. This function is similar to `rescale` but is strictly binned by N x N blocks.
        Also, any numpy functions that accept "axis" argument are supported for reduce functions.

        Parameters
        ----------
        binsize : int, default is 2
            Bin size, such as 2x2.
        method : str or callable, default is numpy.sum
            Reduce function applied to each bin.
        check_edges : bool, default is True
            If True, only divisible `binsize` is accepted. If False, image is cropped at the end to
            match `binsize`.
        dims : str or int, optional
            Spatial dimensions.

        Returns
        -------
        LazyImgArray
            Binned image
        """ 
        if isinstance(method, str):
            binfunc = getattr(np, method)
        elif callable(method):
            binfunc = method
        else:
            raise TypeError("`method` must be a numpy function or callable object.")
               
        shape = []
        scale_ = []
        img_to_reshape = self.img
        for i, a in enumerate(self.axes):
            s = self.shape[i]
            if a in dims:
                b = binsize
                if s % b != 0:
                    if check_edges:
                        raise ValueError(f"Cannot bin axis {a} with length {s} by bin size {binsize}")
                    else:
                        img_to_reshape = img_to_reshape[(slice(None),)*i + (slice(None, s//b*b),)]
            else:
                b = 1
            shape += [s//b, b]
            scale_.append(1/b)
        
        # rechunk to optimize for bin width
        if chunks is None:
            chunks = []
            for a, s, cs in zip(self.axes, self.shape, self.chunksize):
                if a not in dims:
                    chunks.append(cs)
                elif cs % b == 0 and s/cs > 8:
                    chunks.append(cs)
                else:
                    chunks.append((cs//b+1)*b)
                
        img_to_reshape = img_to_reshape.rechunk(chunks=tuple(chunks))
        reshaped_img = img_to_reshape.reshape(tuple(shape))
        axes_to_reduce = tuple(i*2+1 for i in range(self.ndim))
        out = binfunc(reshaped_img, axis=axes_to_reduce)
        out = self.__class__(out)
        out._set_info(self, f"binning(binsize={binsize})")
        out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    def as_uint8(self) -> LazyImgArray:
        img = self.img
        if img.dtype == np.uint8:
            return img
        
        if img.dtype == np.uint16:
            out = img / 256
        elif img.dtype.kind == "f":
            out = img + 0.5
            out = da.clip(out, 0, 255)
        else:
            raise TypeError(f"invalid data type: {img.dtype}")
        out = out.astype(np.uint8)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
    def as_uint16(self) -> LazyImgArray:
        img = self.img
        if img.dtype == np.uint16:
            return img
        if img.dtype == np.uint8:
            out = img * 256
        elif img.dtype == bool:
            out = img
        elif img.dtype.kind == "f":
            out = img + 0.5
            out = da.clip(out, 0, 65535)
        else:
            raise TypeError(f"invalid data type: {img.dtype}")
        out = out.astype(np.uint16)
        out = self.__class__(out)
        out._set_info(self)
        return out
    
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
    