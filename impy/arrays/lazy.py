from __future__ import annotations
import os
from dask import array as da
from dask.diagnostics import ProgressBar
from .imgarray import ImgArray
from .labeledarray import _make_rotated_axis
from .axesmixin import AxesMixin
from .utils._dask_image import *
from .utils._skimage import *
from .utils import _misc, _transform, _structures, _filters, _deconv

from ..utils.deco import *
from ..utils.axesop import *
from ..utils.slicer import *
from ..utils.misc import *
from ..utils.io import *

from .._types import *
from ..axes import ImageAxesError
from .._const import Const

class LazyImgArray(AxesMixin):
    additional_props = ["dirpath", "metadata", "name"]
    def __init__(self, obj: da.core.Array, name:str=None, axes:str=None, dirpath:str=None, 
                 history:list[str]=None, metadata:dict=None):
        if not isinstance(obj, da.core.Array):
            raise TypeError(f"The first input must be dask array, got {type(obj)}")
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
        # Should not be `self.data` because in napari Viewer this function is called every time
        # sliders are moved.
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
    
    
    @same_dtype(asfloat=True)
    def __add__(self, value) -> LazyImgArray:
        if isinstance(value, self.__class__):
            out = self.img + value.img
        else:
            out = self.img + value
        out = self.__class__(out)
        out._set_info(self, next_history="add")
        return out
    
    @same_dtype(asfloat=True)
    def __iadd__(self, value) -> LazyImgArray:
        if isinstance(value, self.__class__):
            self.img += value.img
        else:
            self.img += value
        self.history.append("add")
        return self
    
    @same_dtype(asfloat=True)
    def __sub__(self, value) -> LazyImgArray:
        if isinstance(value, self.__class__):
            out = self.img - value.img
        else:
            out = self.img - value
        out = self.__class__(out)
        out._set_info(self, next_history="subtract")
        return out
    
    @same_dtype(asfloat=True)
    def __isub__(self, value) -> LazyImgArray:
        if isinstance(value, self.__class__):
            self.img -= value.img
        else:
            self.img -= value
        self.history.append("subtract")
        return self
    
    @same_dtype(asfloat=True)
    def __mul__(self, value) -> LazyImgArray:
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            other = value
        elif isinstance(value, self.__class__) and value.dtype.kind != "c":
            value = value.as_float()
            other = value.img
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = value
        out = self.img * other
        out = self.__class__(out)
        out._set_info(self, next_history="multiply")
        return out
    
    @same_dtype(asfloat=True)
    def __imul__(self, value) -> LazyImgArray:
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            other = value
        elif isinstance(value, self.__class__) and value.dtype.kind != "c":
            value = value.as_float()
            other = value.img
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = value
        self.img *= other
        self.history.append("multiply")
        return self
    
    def __truediv__(self, value) -> LazyImgArray:        
        self = self.as_float()
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            value[value==0] = np.inf
            other = value
        elif isinstance(value, self.__class__) and value.dtype.kind != "c":
            value = value.as_float()
            value[value==0] = np.inf
            other = value.img
        elif np.isscalar(value) and value <= 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = value
        out = self.img / other
        out = self.__class__(out)
        out._set_info(self, next_history="divide")
        return out
    
    def __itruediv__(self, value) -> LazyImgArray:
        if self.dtype.kind in "ui":
            raise ValueError("Cannot divide integer inplace.")
        if isinstance(value, np.ndarray) and value.dtype.kind != "c":
            value = value.astype(np.float32)
            value[value==0] = np.inf
            other = value
        elif isinstance(value, self.__class__) and value.dtype.kind != "c":
            value = value.as_float()
            value[value==0] = np.inf
            other = value.img
        elif np.isscalar(value) and value < 0:
            raise ValueError("Cannot multiply negative value.")
        else:
            other = value
        self.img /= other
        self.history.append("divide")
        return self
    
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
        """
        Compute all the task and convert the result into ImgArray. If image size overwhelms MAX_GB
        then MemoryError is raised.
        """        
        if self.gb > Const["MAX_GB"]:
            raise MemoryError(f"Too large: {self.gb:.2f} GB")
        with ProgressBar():
            img = self.img.compute().view(ImgArray)
            for attr in ["name", "dirpath", "axes", "metadata", "history"]:
                setattr(img, attr, getattr(self, attr, None))
        return img
    
    def release(self, update=True) -> LazyImgArray:
        """
        Compute all the task for now and convert to dask again. If image size overwhelms MAX_GB
        then MemoryError is raised.
        """
        if self.gb > Const["MAX_GB"]:
            raise MemoryError(f"Too large: {self.gb:.2f} GB")
        with ProgressBar():
            img = da.from_array(self.img.compute(), chunks=self.chunksize)
            if update:
                self.img = img
                out = self
            else:
                out = self.__class__(img)
                out._set_info(self)
        return out
    
    @dims_to_spatial_axes
    def imsave(self, dirpath:str, dtype=None, *, dims=None):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        if self.metadata is None:
            self.metadata = {}
        if dtype is None:
            dtype = self.dtype
        
        self = self.as_img_type(dtype).sort_axes()
        imsave_kwargs = get_imsave_meta_from_img(self, update_lut=False)
        
        rechunk_to = switch_slice(dims, self.axes, ifin=self.shape, ifnot=1)
        
        img = self.img.rechunk(rechunk_to)
        
        # convert to float32 if image is float64
        if img.dtype == np.float64:
            img = img.astype(np.float32)
        
        # save image
        def _imwrite(arr, block_info=None, **kwargs):
            if block_info is None:
                return None
            path = os.path.join(dirpath, "-".join(map(str, block_info[0]["chunk-location"])) + ".tif")
            imwrite(path, arr, **kwargs)
            return arr
        
        da.map_blocks(_imwrite, img, dtype=img.dtype, **imsave_kwargs).compute()
        print(f"Succesfully saved: {dirpath}")
        return None
    
    def rechunk(self, chunks="auto", *, threshold=None, block_size_limit=None, balance=False, update=False) -> LazyImgArray:
        """
        Rechunk the bound dask array.

        Parameters
        ----------
        chunks, threshold, block_size_limit, balance
            Passed directly to dask.array's rechunk

        Returns
        -------
        LazyImgArray
            Rechunked dask array is bound. History will not be updated.
        """        
        rechunked = self.img.rechunk(chunks=chunks, threshold=threshold, 
                                     block_size_limit=block_size_limit, balance=balance)
        if update:
            self.img = rechunked
            return self
        else:
            out = self.__class__(rechunked)
            out._set_info(self)
            return out
        
    def apply_dask_func(self, funcname:str, *args, **kwargs) -> LazyImgArray:
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
    
    def apply(self, func, c_axes:str=None, drop_axis:Iterable[int]=[], new_axis:Iterable[int]=None, 
              dtype=np.float32, rechunk_to:tuple[int,...]|str="none", dask_wrap:bool=False,
              args:tuple=None, kwargs:dict[str]=None) -> LazyImgArray:
        """
        Rechunk array in a correct shape and apply function using `map_blocks`. This function is similar
        to the `apply_dask` function in `MetaArray` while returns dask array bound LazyImgArray.

        Parameters
        ----------
        func : callable
            Function to apply for each chunk.
        c_axes : str, optional
            Axes to iterate.
        drop_axis : Iterable[int], optional
            Passed to map_blocks.
        new_axis : Iterable[int], optional
            Passed to map_blocks.
        dtype : any that can be converted to np.dtype object, default is np.float32
            Output data type.
        rechunk_to : tuple[int,...], optional
            In what size input array should be rechunked before `map_blocks` iteration. If str is given, 
            array will be rechunked in following rules:
                - "none": No rechunking
                - "default": Rechunked with "auto" method for each spatial dimension.
                - "max": Rechunked to the shape size for each spatial dimension.
        dask_wrap : bool, optional
            If True, for each chunk array will be converted to dask and rechunked with "auto" option
            before function call.
        args : tuple, optional
            Arguments that will passed to `func`.
        kwargs : dict
            Keyword arguments that will passed to `func`.

        Returns
        -------
        LazyImgArray
            Dask array after function is applied is bound to this newly generated object.
        """        
        slice_in = []
        slice_out = []
        for i, a in enumerate(self.axes):
            if a in c_axes:
                slice_in.append(0)
                slice_out.append(np.newaxis)
            else:
                slice_in.append(slice(None))
                slice_out.append(slice(None))
                
        slice_in = tuple(slice_in)
        slice_out = tuple(slice_out)
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = dict()
            
        if rechunk_to == "none":
            input_ = self.img
        else:
            if rechunk_to == "default":
                rechunk_to = switch_slice(c_axes, self.axes, ifin=1, ifnot="auto")
            
            elif rechunk_to == "max":
                rechunk_to = switch_slice(c_axes, self.axes, ifin=1, ifnot=self.shape)
                
            input_ = self.img.rechunk(rechunk_to)
        
        if dask_wrap:
            def _func(arr, *args, **kwargs):
                out = func(da.from_array(arr[slice_in]), *args, **kwargs)
                return out[slice_out].compute()
        else:
            def _func(arr, *args, **kwargs):
                out = func(arr[slice_in], *args, **kwargs)
                return out[slice_out]
        
        out = da.map_blocks(_func, input_, *args, drop_axis=drop_axis, new_axis=new_axis, 
                            dtype=dtype, **kwargs)
        
        out = self.__class__(out)
        out._set_info(self, make_history(func.__name__, args, kwargs), new_axes=self.axes)
        return out
    
    def rotated_crop(self, origin, dst1, dst2, dims="yx") -> LazyImgArray:
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
        
        # Because output shape changes, we have to tell dask what chunk size it should be, otherwise output
        # shape is estimated in a wrong way. 
        output_chunks = list(self.chunksize)
        for i, a in enumerate(dims):
            it = self.axisof(a)
            output_chunks[it] = all_coords.shape[i+1]
            
        cropped_img = self.apply(ndi.map_coordinates, 
                                 c_axes=complement_axes(dims, self.axes), 
                                 dtype=self.dtype,
                                 rechunk_to="max",
                                 args=(all_coords,),
                                 kwargs=dict(prefilter=False, order=1, chunks=output_chunks)
                                 )

        cropped_img._set_info(self, "rotated_crop")
        return cropped_img
    
    def _switch_apply(self, func_dask, func_default, dims, args=None, kwargs=None):
        try:
            self.try_max_chunk(dims)
            func = func_default
            rechunk_to = "none"
            dask_wrap = False
        except MemoryError:
            func = func_dask
            rechunk_to = "max"
            dask_wrap = True
            
        return self.apply(func,
                          c_axes=complement_axes(dims, self.axes),
                          rechunk_to=rechunk_to,
                          dask_wrap=dask_wrap,
                          args=args,
                          kwargs=kwargs
                          )
    
    @dims_to_spatial_axes
    def gaussian_filter(self, sigma:nDFloat=1.0, *, dims=None) -> LazyImgArray:
        return self._switch_apply(dafil.gaussian_filter,
                                  _filters.gaussian_filter,
                                  dims=dims,
                                  args=(sigma,)
                                  )
    
    @dims_to_spatial_axes
    def median_filter(self, radius:float=1, *, dims=None) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        return self._switch_apply(dafil.median_filter,
                                  _filters.median_filter,
                                  dims=dims,
                                  kwargs=dict(footprint=disk)
                                  )
    
    @dims_to_spatial_axes
    def mean_filter(self, radius:float=1, *, dims=None) -> LazyImgArray:
        disk = _structures.ball_like(radius, len(dims))
        kernel = disk/np.sum(disk)
        return self._switch_apply(dafil.convolve,
                                  _filters.convolve,
                                  dims=dims,
                                  args=(kernel,),
                                  )
    
    @dims_to_spatial_axes
    def convolve(self, kernel, *, mode:str="reflect", cval:float=0, dims=None) -> LazyImgArray:
        return self._switch_apply(dafil.convolve,
                                  _filters.convolve,
                                  dims=dims,
                                  args=(kernel,),
                                  kwargs=dict(mode=mode, cval=cval)
                                  )
    
    @dims_to_spatial_axes
    def edge_filter(self, method:str="sobel", *, dims=None) -> LazyImgArray:
        f = {"sobel": dafil.sobel,
             "prewitt": dafil.prewitt}[method]
        return self.apply(f, 
                          c_axes=complement_axes(dims, self.axes), 
                          dtype=self.dtype,
                          rechunk_to="max",
                          dask_wrap=True
                          )
    
    @dims_to_spatial_axes
    def affine(self, matrix=None, scale=None, rotation=None, shear=None, translation=None, *,
               order=1, dims=None) -> LazyImgArray:
        if matrix is None:
            matrix = _transform.compose_affine_matrix(scale=scale, rotation=rotation, 
                                                      shear=shear, translation=translation,
                                                      ndim=len(dims))
        return self._switch_apply(daintr.affine_transform,
                                  _transform.warp,
                                  dims=dims,
                                  kwargs=dict(matrix=matrix, order=order)
                                  )
    
    @dims_to_spatial_axes
    def fft(self, *, shape="same", shift:bool=True, dims=None) -> LazyImgArray:
        axes = [self.axisof(a) for a in dims]
        if shape == "square":
            s = 2**int(np.ceil(np.max(self.sizesof(dims))))
            shape = (s,) * len(dims)
        elif shape == "same":
            shape = None
        else:
            shape = check_nd(shape, len(dims))
        freq = da.fft.rfftn(self.img.astype(np.float32), s=shape, axes=axes).astype(np.complex64)
        if shift:
            freq[:] = da.fft.fftshift(freq)
        out = self.__class__(freq)
        out._set_info(self, "fft")
        return out

    @dims_to_spatial_axes
    def ifft(self, real:bool=True, *, shift:bool=True, dims=None) -> LazyImgArray:
        if shift:
            freq = da.fft.ifftshift(self.img)
        else:
            freq = self.img
        out = da.fft.irfftn(freq, axes=[self.axisof(a) for a in dims]).astype(np.complex64)
        
        if real:
            out = da.real(out)
        
        out = self.__class__(out)
        out._set_info(self, "ifft")
        return out
    
    def chunksizeof(self, axis:str):
        return self.img.chunksize[self.axes.find(axis)]
    
    def chunksizesof(self, axes:str):
        return tuple(self.chunksizeof(a) for a in axes)
    
    def try_max_chunk(self, dims):
        new_chunks = switch_slice(dims, self.axes, ifin=self.shape, ifnot=1)
        gb_per_chunk = np.prod(new_chunks) * self.itemsize * 1e-9
        if gb_per_chunk > Const["MAX_GB"]:
            raise MemoryError(f"Cannot allocate {gb_per_chunk} GB for one chunk.")
        else:
            self.rechunk(chunks=new_chunks, update=True)
        return self
        
    def transpose(self, axes):
        if self.axes.is_none():
            new_axes = None
        else:
            new_axes = "".join([self.axes[i] for i in list(axes)])
        out = self.__class__(self.img.transpose(axes))
        out._set_info(self, new_axes=new_axes)
        return out
        
    def sort_axes(self):
        """
        Sort image dimensions to ptzcyx-order

        Returns
        -------
        MetaArray
            Sorted image
        """
        order = self.axes.argsort()
        return self.transpose(tuple(order))
    
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
            chunks = switch_slice(axis, self.axes, ifin=np.maximum(self.shape, 2048), ifnot=["auto"]*self.ndim)
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
        
        if binsize == 1:
            return self
        
        img_to_reshape, shape, scale_ = _misc.adjust_bin(self.img, binsize, check_edges, dims, self.axes)
        # rechunk to optimize for bin width
        if chunks is None:
            chunks = []
            for a, s, cs in zip(self.axes, self.shape, self.chunksize):
                if a not in dims:
                    chunks.append(cs)
                elif cs % binsize == 0 and s/cs > 8:
                    chunks.append(cs)
                else:
                    chunks.append((cs//binsize+1)*binsize)
                
        img_to_reshape = img_to_reshape.rechunk(chunks=tuple(chunks))
        reshaped_img = img_to_reshape.reshape(shape)
        axes_to_reduce = tuple(i*2+1 for i in range(self.ndim))
        out = binfunc(reshaped_img, axis=axes_to_reduce)
        out = self.__class__(out)
        out._set_info(self, f"binning(binsize={binsize})")
        out.axes = str(self.axes) # _set_info does not pass copy so new axes must be defined here.
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)})
        return out
    
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def lucy(self, psf:np.ndarray, niter:int=50, eps:float=1e-5, *, dims=None, 
             update:bool=False) -> LazyImgArray:
        
        psf_ft, psf_ft_conj = _deconv.check_psf(self, psf, dims)

        return self.apply(_deconv.richardson_lucy, 
                          c_axes=complement_axes(dims, self.axes),
                          rechunk_to="max",
                          args=(psf_ft, psf_ft_conj, niter, eps)
                          )
    
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