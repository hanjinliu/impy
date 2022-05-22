from __future__ import annotations
import warnings
import numpy as np
from numpy.typing import DTypeLike
import os
from pathlib import Path
import itertools
from functools import partial
import inspect
from typing import TYPE_CHECKING, Any, Hashable, TypeVar
from warnings import warn
from scipy import ndimage as ndi

from .specials import PropArray
from ._utils._skimage import skmes, skseg
from ._utils import _misc, _docs
from .bases import MetaArray
from .label import Label

from ..utils.misc import check_nd, largest_zeros
from ..utils.axesop import complement_axes, find_first_appeared
from ..utils.deco import check_input_and_output, dims_to_spatial_axes
from ..utils.io import IO

from ..collections import DataList
from ..axes import ImageAxesError
from .._types import Dims, nDInt, nDFloat, Callable, Coords, Iterable

if TYPE_CHECKING:
    from typing_extensions import Self

Ax = TypeVar("Ax", bound=Hashable)

class LabeledArray(MetaArray[Ax]):
    _name: str
    _source: Path | None
    _metadata: dict[str, Any]
    _labels: Label | None
    
    def __new__(
        cls: type[LabeledArray], 
        obj,
        name: str | None = None,
        axes: Iterable[Ax] | None = None,
        source: str | Path | None = None, 
        metadata: dict[str, Any] | None = None,
        dtype: DTypeLike = None,
    ) -> Self:
        self: LabeledArray = super().__new__(
            cls, obj, name, axes, source, metadata, dtype
        )
        self._labels = None
        return self

    @property
    def range(self) -> tuple[float, float]:
        """Return min/max range of the array."""
        return self.min(), self.max()
    
    @property
    def labels(self) -> Label | None:
        return self._labels
    
    @labels.setter
    def labels(self, value: np.ndarray | None):
        if value is None:
            self._labels = None
            return

        if value is self:
            raise ValueError("Setting labels recursively is not allowed.")
        
        if not isinstance(value, Label):
            # convert input
            arr = np.asarray(value)
            if arr.dtype.kind == "i":
                arr = arr.astype(np.uint64)
            elif arr.dtype.kind != "u":
                raise TypeError(
                    f"Input label must be unsigned int but has wrong dtype {arr.dtype}."
                )
            axes = str(self.axes)[-arr.ndim:]
            value = Label(arr, axes=axes).optimize()
        
        
        if not _shape_match(self, value):
            raise ValueError(
                f"Shape of input label ({value.shape_info}) does not match the "
                f"parent array ({self.shape_info})."
            )
        self._labels = value
    
    @labels.deleter
    def labels(self):
        self._labels = None
    
    def set_scale(self, other=None, **kwargs) -> None:
        super().set_scale(other, **kwargs)
        if self.labels is not None:
            self.labels.set_scale(other, **kwargs)
        return None
        
        
    def _repr_dict_(self):
        if self.labels is not None:
            labels_shape_info = self.labels.shape_info
        else:
            labels_shape_info = "No label"
        return {
            "name": self.name,
            "shape": self.shape_info,
            "label shape ": labels_shape_info,
            "dtype": self.dtype,
            "source": self.source,
            "scale": self.scale,
        }
    
    
    def imsave(self, save_path: str, dtype: DTypeLike = None):
        """
        Save image at the same directory as the original image by default. For tif file format, if the
        image contains wrong axes for ImageJ (= except for tzcyx), then it will converted automatically 
        if possible. For mrc file format, only zyx and yx is allowed. zyx-scale is also saved.

        Parameters
        ----------
        save_path : str
            File name.
        dtype : dtype-like, optional
            In what data type img will be saved.
        
        Returns
        -------
        None
        """        
        save_path = str(save_path)
        _, ext = os.path.splitext(save_path)
        
        if ext == "":
            if self.source is not None:
                ext = self.source.suffix
            else:
                ext = ".tif"
            save_path += ext
    
        if os.sep not in save_path:
            if self.source is None:
                raise ValueError(
                    "Image directory path is unknown. Set by \n"
                    " >>> img.source = \"...\"\n"
                    "or specify absolute path like\n"
                    " >>> img.imsave(\"/path/to/XXX.tif\")"
                    )
            save_path = os.path.join(self.source.parent, save_path)
        if self.metadata is None:
            self.metadata = {}
        if dtype is None:
            dtype = self.dtype
            
        # save image
        IO.imsave(save_path, self)

        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Basic Functions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def __array_finalize__(self, obj):
        self._labels = getattr(self, "_labels", None)
        super().__array_finalize__(obj)
        if isinstance(obj, LabeledArray):
            self._view_labels(obj)
    
    def _set_info(self, other: Self, new_axes: Any = MetaArray._INHERIT):
        self._labels = getattr(self, "_labels", None)
        super()._set_info(other, new_axes)
        if isinstance(other, LabeledArray):
            self._view_labels(other)
    
    def _view_labels(self, other: Self):
        """Make a view of label **if possible**."""
        if (
            other.labels is not None and
            self.axes.contains(other.labels.axes) and
            _shape_match(self, other.labels)
        ):
            if self is not other:
                self.labels = other.labels.copy()
            else:
                self.labels = other.labels
        
    
    def _getitem_additional_set_info(self, other: Self, **kwargs):
        super()._getitem_additional_set_info(other, **kwargs)
        key = kwargs["key"]
        if other.axes and other.labels is not None and not isinstance(key, np.ndarray):
            if isinstance(key, tuple):
                _keys = key
            else:
                _keys = (key,)
            label_sl = [_keys[i] for i, a in enumerate(other.axes) 
                        if a in other.labels.axes and i < len(_keys)]
                    
            if len(label_sl) == 0 or len(label_sl) > other.labels.ndim:
                label_sl = (slice(None),)
            try:
                self.labels = other.labels[tuple(label_sl)]
            except IndexError as e:
                warn(f"Labels was not inherited due to IndexError : {e}", UserWarning)
        
        return None
    
    def _update(self, out: Self):
        self.value[:] = out.as_img_type(self.dtype).value[:]
        return None
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Type Conversions
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    def as_uint8(self) -> Self:
        if self.dtype == np.uint8:
            return self
        
        if self.dtype == np.uint16:
            out = self.value / 256
        elif self.dtype == bool:
            out = self.value
        elif self.dtype.kind == "f":
            out = self.value + 0.5
            out[out<0] = 0
            out[out>255] = 255
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out = out.astype(np.uint8)
        out = out.view(self.__class__)
        out._set_info(self)
        return out


    def as_uint16(self) -> Self:
        if self.dtype == np.uint16:
            return self
        if self.dtype == np.uint8:
            out = self.value.astype(np.uint16) * 256
        elif self.dtype == bool:
            out = self.value
        elif self.dtype.kind == "f":
            out = self.value + 0.5
            out[out<0] = 0
            out[out>65535] = 65535
        else:
            raise TypeError(f"invalid data type: {self.dtype}")
        out = out.astype(np.uint16)
        out = out.view(self.__class__)
        out._set_info(self)
        return out
    
    def as_float(self) -> Self:
        if self.dtype == np.float32:
            return self
        out = self.value.astype(np.float32).view(self.__class__)
        out._set_info(self)
        return out
        
    
    def as_img_type(self, dtype=np.uint16) -> Self:
        dtype = np.dtype(dtype)
        if self.dtype == dtype:
            return self
        elif dtype == "uint16":
            return self.as_uint16()
        elif dtype == "uint8":
            return self.as_uint8()
        elif dtype == "float32":
            return self.as_float()
        elif dtype == "bool":
            return self.astype("bool")
        elif dtype == "float64":
            warn("Data type float64 is not valid for images. It was converted to float32 instead",
                 UserWarning)
            return self.as_float()
        elif dtype == "complex64":
            return self.astype(np.complex64)
        elif dtype == "complex128":
            warn("Data type complex128 is not valid for images. It was converted to complex64 instead",
                 UserWarning)
            return self.astype(np.complex64)
        elif dtype in ("int8", "int16"):
            return self.astype(dtype)
        else:
            raise ValueError(f"dtype: {dtype}")
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Simple Visualizations
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    def hist(self, contrast=None):
        """Show intensity profile."""
        from ._utils import _plot as _plt
        _plt.hist(self.value, contrast)

        return None

    @dims_to_spatial_axes
    def imshow(self, label: bool = False, dims = 2, alpha=0.3, **kwargs):
        from ._utils import _plot as _plt
        if label and self.labels is None:
            label = False
        if self.ndim == 1:
            _plt.plot_1d(self.value, **kwargs)
        elif self.ndim == 2:
            if label:
                _plt.plot_2d_label(self.value, self.labels.value, alpha, **kwargs)
            else:
                _plt.plot_2d(self.value, **kwargs)
            self.hist()
            
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = self.split(axis=find_first_appeared(self.axes, include=self.axes, exclude=dims))
                if len(imglist) > 24:
                    warnings.warn(
                        "Too many images. First 24 images are shown.",
                        UserWarning,
                    )
                    imglist = imglist[:24]
                if label:
                    _plt.plot_3d_label(imglist.value, imglist.labels.value, alpha, **kwargs)
                else:
                    _plt.plot_3d(imglist, **kwargs)

            else:
                n_chn = self.shape.c
                fig, ax = _plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
                for i in range(n_chn):
                    img = self[f"c={i}"]
                    if label:
                        _plt.plot_2d_label(img.value, img.labels.value, alpha, ax[i], **kwargs)
                    else:
                        _plt.plot_2d(img.value, ax=ax[i], **kwargs)
        else:
            raise ValueError("Image must have three or less dimensions.")
        
        _plt.show()

        return self

    def imshow_comparewith(self, other: Self, **kwargs):
        from ._utils import _plot as _plt
        fig, ax = _plt.subplots(1, 2, figsize=(8, 4))
        _plt.plot_2d(self.value, ax=ax[0], **kwargs)
        _plt.plot_2d(other.value, ax=ax[1], **kwargs)        
        _plt.show()
        return self
    
    @dims_to_spatial_axes
    def imshow_label(self, alpha=0.3, dims=2, **kwargs):
        from ._utils import _plot as _plt
        if not self.labels is not None:
            raise AttributeError("No label to show.")
        if self.ndim == 2:
            _plt.plot_2d_label(self.value, self.labels.value, alpha, **kwargs)
            self.hist()
        elif self.ndim == 3:
            if "c" not in self.axes:
                imglist = self.split(axis=find_first_appeared(self.axes, include=self.axes, exclude=dims))
                if len(imglist) > 24:
                    print("Too many images. First 24 images are shown.")
                    imglist = imglist[:24]

                _plt.plot_3d_label(imglist.value, imglist.labels.value, alpha, **kwargs)

            else:
                n_chn = self.shape.c
                fig, ax = _plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
                for i in range(n_chn):
                    img = self[f"c={i}"]
                    _plt.plot_2d_label(img.value, img.labels.value, alpha, ax[i], **kwargs)
                    
        else:
            raise ValueError("Image must be two or three dimensional.")
        
        _plt.show()
        return self
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Cropping
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @_docs.write_docs
    @check_input_and_output
    @dims_to_spatial_axes
    def crop_center(self, scale: nDFloat = 0.5, *, dims=2) -> Self:
        r"""
        Crop out the center of an image. 
        
        Parameters
        ----------
        scale : float or array-like, default is 0.5
            Scale of the cropped image. If an array is given, each axis will be cropped in different scales,
            using each value respectively.
        {dims}
        
        Returns
        -------
        Self
            CroppedImage
            
        Examples
        --------
        1. Create a :math:`512\times512` image from a :math:`1024\times1024` image.
        
            >>> img_cropped = img.crop_center(scale=0.5)
            
        2. Create a :math:`21\times256\times256` image from a :math:`63\times1024\times1024` image.
        
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
    
    @check_input_and_output
    def crop_kernel(self, radius:nDInt=2) -> Self:
        r"""
        Make a kernel from an image by cropping out the center region. This function is useful especially
        in `ImgArray.defocus()`.

        Parameters
        ----------
        radius : int or array-like of int, default is 2
            Radius of the kernel.

        Returns
        -------
        LabeledArray
            Kernel
        
        Examples
        --------
        Make a :math:`4\times4\times4` kernel from a point spread function image (suppose the image shapes 
        are all even numbers).
            
            >>> psf = ip.imread(r".../PSF.tif")
            >>> psfker = psf.crop_kernel()
            >>> psfer.shape
            (4, 4, 4)
        """        
        sizes = self.shape
        radii = check_nd(radius, len(sizes))
        return self[tuple(slice(s//2-r, (s+1)//2+r) for s, r in zip(sizes, radii))]
    
    @_docs.write_docs
    @check_input_and_output
    @dims_to_spatial_axes
    def remove_edges(self, pixel:nDInt=1, *, dims=2) -> Self:
        """
        Remove pixels from the edges.

        Parameters
        ----------
        pixel : int or array-like, default is 1
            Number of pixels to remove. If an array is given, each axis will be cropped with different pixels,
            using each value respectively.
        {dims}

        Returns
        -------
        LabeledArray
            Cropped image.
        """        
        if hasattr(pixel, "__iter__") and len(pixel) == 3 and len(dims) == 2:
            dims = "zyx"
        pixel = np.asarray(check_nd(pixel, len(dims)), dtype=np.int64)
        if np.any(pixel < 0):
            raise ValueError("`pixel` must be positive.")
            
        slices = []
        for a, px in zip(dims, pixel):
            slices.append(f"{a}={px}:-{px}")
        
        out = self[";".join(slices)]
        return out
    
    @_docs.write_docs
    @check_input_and_output
    @dims_to_spatial_axes
    def rotated_crop(self, origin, dst1, dst2, dims=2) -> Self:
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
        dst2 :(float, float)
        {dims}
        
        Returns
        -------
        LabeledArray
            Cropped array.
        """
        origin = np.asarray(origin)
        dst1 = np.asarray(dst1)
        dst2 = np.asarray(dst2)
        ax0 = _misc.make_rotated_axis(origin, dst2)
        ax1 = _misc.make_rotated_axis(dst1, origin)
        all_coords = ax0[:, np.newaxis] + ax1[np.newaxis] - origin
        all_coords = np.moveaxis(all_coords, -1, 0)
        cropped_img = self._apply_dask(
            ndi.map_coordinates, complement_axes(dims, self.axes), 
            dtype=self.dtype,
            args=(all_coords,),
            kwargs=dict(prefilter=False, order=1)
        )
        cropped_img = cropped_img.view(self.__class__)
        cropped_img.axes = self.axes
        if self.labels is not None:
            try:
                lbl = self.labels
                cropped_labels = np.empty(lbl.shape[:-2] + all_coords.shape[1:], dtype=lbl.dtype)
                for sl, lbl2d in lbl.iter(complement_axes(dims, lbl.axes)):
                    cropped_labels[sl] = ndi.map_coordinates(lbl2d, all_coords, prefilter=False, order=0)
            except Exception:
                print("cropping labels failed")
            else:
                cropped_img.append_label(cropped_labels)
        
        return cropped_img
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Label handling and others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    
    @_docs.write_docs
    @dims_to_spatial_axes
    def specify(self, center: Coords, radius: Coords, *, dims: Dims = None, 
                labeltype: str = "square") -> Label:
        """
        Make rectangle or ellipse labels from points.
        
        Parameters
        ----------
        center : array like or MarkerFrame
            Coordinates of centers. For MarkerFrame, it must have the same axes order.
        radius : float or array-like
            Radius of labels.
        {dims}
        labeltype : str, default is "square"
            The shape of labels.

        Returns
        -------
        Label
            Labeled regions.
        
        Examples
        --------
        Find single molecules, draw circular labels around them if mean values were greater than 100.
            >>> coords = img.find_sm()
            >>> filter_func = lambda a: np.mean(a) > 100
            >>> img.specify(coords, 3.5, filt=filter_func, labeltype="circle")
            >>> ip.gui.add(img)
        """
        from ..frame import MarkerFrame
        if isinstance(center, MarkerFrame):
            from ._utils._process_numba import _specify_circ_2d, _specify_circ_3d, _specify_square_2d, _specify_square_3d
            ndim = len(dims)
            radius = np.asarray(check_nd(radius, ndim), dtype=np.float32)
            
            if labeltype in ("square", "s"):
                radius = radius.astype(np.uint8)
                _specify = {2: _specify_square_2d,
                            3: _specify_square_3d}[ndim]
                
            elif labeltype in ("circle", "c"):
                _specify = {2: _specify_circ_2d,
                            3: _specify_circ_3d}[ndim]
            
            else:
                raise ValueError("`labeltype` must be 'square' or 'circle'.")
            
            label_axes = str(center.col_axes)
            label_shape = self.sizesof(label_axes)
            labels = largest_zeros(label_shape)
            
            n_label = 1
            for sl, crds in center.iter(complement_axes(dims, center.col_axes)):
                _specify(labels[sl], crds.values, radius, n_label)
                n_label += len(crds)
        
            if self.labels is not None:
                print("Existing labels are updated.")
            self.labels = Label(labels, axes=label_axes).optimize()
            self.labels.set_scale(self)
        
        else:
            center = np.asarray(center)
            if center.ndim == 1:
                center = center.reshape(1, -1)
            
            cols = {2:"yx", 3:"zyx"}[center.shape[1]]
            center = MarkerFrame(center, columns=cols, dtype=np.uint16)

            return self.specify(center, radius, dims=dims, labeltype=labeltype)     
        
        return self.labels
       
    @_docs.write_docs
    @check_input_and_output
    def reslice(self, a, b=None, *, order: int = 1) -> PropArray:
        """
        Measure line profile (kymograph) iteratively for every slice of image. This function is almost 
        same as `skimage.measure.profile_line`, but can reslice 3D-images. The argument `linewidth` is 
        not implemented here because it is useless.

        Parameters
        ----------
        a : array-like
            Path or source coordinate. If the former, it must be like:
            `a = [[y0, x0], [y1, x1], ..., [yn, xn]]`
        b : array-like, optional
            Destination coordinate. If specified, `a` must be the source coordinate.
        {order}

        Returns
        -------
        PropArray
            Line scans.
        
        Examples
        --------
        1. Rescile along a line and fit to a model function for every time frame.
        
            >>> scan = img.reslice([18, 32], [53, 48])
            >>> out = scan.curve_fit(func, init, return_fit=True)
            >>> plt.plot(scan[0])
            >>> plt.plot(out.fit[0])
            
        2. Rescile along a path.
        
            >>> scan = img.reslice([[18,32], [53,48], [22,45], [28, 32]])
        """        
        # path = [[y1, x1],[y2, x2], ..., [yn, xn]]
        if b is not None:
            a = [list(a), list(b)]
        a = np.asarray(a, dtype=np.float32)
        npoints, ndim = a.shape
        
        if npoints < 2:
            raise ValueError("Insufficient number of points for a path.")
        elif npoints == 2:
            src, dst = a
            d = dst - src
            length = int(np.ceil(np.sqrt(np.sum(d**2)) + 1))
            coords = np.vstack([np.linspace(src_, dst_, length) for src_, dst_ in zip(src, dst)])
        else:    
            from ._utils._process_numba import _get_coordinate
            
            each_length = np.sqrt(np.sum(np.diff(a, axis=0)**2, axis=1))
            total_length = np.sum(each_length)
            
            coords = np.zeros((ndim, int(total_length)+1))
            _get_coordinate(a, coords)
        
        coords = coords[(slice(None),)+(np.newaxis,)*(ndim-1)]
        
        if ndim == self.ndim:
            dims = self.axes
        else:
            dims = complement_axes("c", self.axes)[-ndim:]
        c_axes = complement_axes(dims, self.axes)
        
        result = self.as_float()._apply_dask(
            ndi.map_coordinates, 
            c_axes=c_axes, 
            dtype=self.dtype,
            drop_axis=-1,
            args=(coords,),
            kwargs=dict(prefilter=order>1, order=order)
        )
        
        sl = [slice(None)]*result.ndim
        for a in dims[:-1]:
            i = self.axisof(a)
            sl[i] = 0
        
        out = PropArray(result[tuple(sl)], name=self.name, dtype=np.float32,
                        axes=c_axes+dims[-1], propname="reslice")
        
        out.set_scale(self)
        return out
    
    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def label(
        self,
        ref_image: np.ndarray | None = None,
        *,
        dims: Dims = None,
        connectivity: int | None = None
    ) -> Label:
        """
        Label image using skimage's label().

        Parameters
        ----------
        ref_image : array, optional
            Image to make label, by default self is used.
        {dims}
        {connectivity}

        Returns
        -------
        Label
            Labeled image.
        
        Examples
        --------
        Label the image with threshold and visualize with napari.
        
            >>> thr = img.threshold()
            >>> img.label(thr)
            >>> ip.gui.add(img)
        """        
        # check the shape of label_image
        if ref_image is None:
            ref_image = self
        else:
            if not isinstance(ref_image, MetaArray):
                ref_image = MetaArray(
                    np.asarray(ref_image),
                    axes=self.axes[-self.ndim:]
                )
            if not self.axes.contains(ref_image.axes):
                raise ImageAxesError(
                    "Not all the axes in `ref_image` are included in self: "
                    f"{ref_image.axes} and {self.axes}"
                )
            elif not _shape_match(self, ref_image):
                raise ImageAxesError("Shape mismatch.")
        
        c_axes = complement_axes(dims, self.axes)
        labels = largest_zeros(ref_image.shape)
        
        labels[:] = ref_image._apply_dask(
            skmes.label, 
            c_axes=c_axes, 
            kwargs=dict(background=0, connectivity=connectivity)
        ).view(np.ndarray)
    
        # correct the label numbers of `labels`
        labels = labels.view(Label)
        labels._set_info(ref_image)
        labels = labels.increment_iter(c_axes).optimize()
        labels.set_scale(self)
        self.labels = labels
        return self.labels
    
    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def label_if(
        self,
        ref_image: np.ndarray | None = None,
        filt: Callable[..., bool] | None = None,
        *,
        dims: Dims = None,
        connectivity: int | None = None,
    ) -> Label:
        """
        Label image if the region satisfies the condition.
        
        Label image using `ref_image` as reference image only if certain 
        condition dictated in `filt` is satisfied. `regionprops_table` is called
        inside every time image is labeled.
        
            .. code-block:: python
            
                def filt(img, lbl, area, major_axis_length):
                    return area>10 and major_axis_length>5

        Parameters
        ----------
        ref_image : array, optional
            Image to make label, by default self is used.
        filt : callable, positional argument but not optional
            Filter function. The first argument is intensity image sliced from 
            `self`, the second is label image sliced from labeled `ref_image`,
            and the rest arguments is properties that will be calculated using
            `regionprops` function. The property arguments **must be named
            exactly same** as the properties in `regionprops`. Number of 
            arguments can be two.
        {dims}
        {connectivity}

        Returns
        -------
        LabeledArray
            Labeled image
        
        Example
        -------
        1. Label regions if only intensity is high.
            >>> def high_intensity(img, lbl, slice):
            >>>     return np.mean(img[slice]) > 10000
            >>> img.label_if(lbl, filt)
        
        2. Label regions if no hole exists.
            >>> def no_hole(img, lbl, euler_number):
            >>>     return euler_number > 0
            >>> img.label_if(lbl, filt)
        
        3. Label regions if centroids are inside themselves.
            >>> def no_hole(img, lbl, centroid):
            >>>     yc, xc = map(int, centroid)
            >>>     return lbl[yc, xc] > 0
            >>> img.label_if(lbl, filt)
        """        
        import pandas as pd
        # check the shape of ref_image
        if ref_image is None:
            ref_image = self
        
        else:
            if not isinstance(ref_image, MetaArray):
                ref_image = MetaArray(
                    np.asarray(ref_image),
                    axes=str(self.axes)[-self.ndim:]
                )
            if not self.axes.contains(ref_image.axes):
                raise ImageAxesError(
                    "Not all the axes in `ref_image` are included in self: "
                    f"{ref_image.axes} and {self.axes}"
                )
            elif not _shape_match(self, ref_image):
                raise ImageAxesError("Shape mismatch.")
        
        # check filter function
        if filt is None:
            raise ValueError("`filt` must be given.")
        if not callable(filt):
            raise TypeError("`filt` must be callable.")
        
        properties = tuple(inspect.signature(filt).parameters)[2:]
            
        c_axes = complement_axes(dims, self.axes)
        labels = largest_zeros(ref_image.shape)
        offset = 1
        for sl, lbl in ref_image.iter(c_axes):
            lbl = skmes.label(lbl, background=0, connectivity=connectivity)
            img = self.value[sl]
            # Following lines are essentially doing the same thing as 
            # `skmes.regionprops_table`. However, `skmes.regionprops_table`
            # returns tuples in the separated columns in DataFrame and rename
            # property names like "centroid-0" and "centroid-1".
            props_obj = skmes.regionprops(lbl, img, cache=False)
            d = {prop_name: [getattr(prop, prop_name) for prop in props_obj]
                 for prop_name in properties}
            df = pd.DataFrame(d)
            del_list = [i+1 for i, r in df.iterrows() if not filt(img, lbl, **r)]
            labels[sl] = skseg.relabel_sequential(
                np.where(np.isin(lbl, del_list), 0, lbl),
                offset=offset
            )[0]
            offset += labels.max()
        
        labels = labels.view(Label).optimize()
        labels._set_info(ref_image)
        labels.set_scale(self)
        self.labels = labels
        return self.labels
    
    @check_input_and_output
    def append_label(self, label_image: np.ndarray, new: bool = False) -> Label:
        """
        Append new labels from an array. This function works for boolean or signed int arrays.

        Parameters
        ----------
        label_image : np.ndarray
            Labeled image.
        new : bool, default is False
            If True, existing labels will be removed anyway.
        
        Returns
        -------
        Label
            New labels.
        
        Example
        -------
        Make label from different channels.
        
            >>> thr0 = img["c=0"].threshold("90%")
            >>> thr0.label() # binary to label
            >>> thr1 = img["c=1"].threshold("90%")
            >>> thr1.label() # binary to label
            >>> img.append_label(thr0.labels)
            >>> img.append_label(thr1.labels)
            
        If `thr0` has 100 labels and `thr1` has 150 labels then `img` will have :math:`100+150=250` labels.
        """
        # check and cast label dtype
        if not isinstance(label_image, np.ndarray):
            raise TypeError(f"`label_image` must be ndarray, but got {type(label_image)}")
        elif label_image.dtype.kind == "u":
            pass
        elif label_image.dtype == bool:
            label_image = label_image.astype(np.uint8)
        elif label_image.dtype == np.int32:
            label_image = label_image.astype(np.uint16)
        elif label_image.dtype == np.int64:
            label_image = label_image.astype(np.uint32)
        elif label_image.dtype.kind == "i":
            label_image = label_image.astype(np.uint8)
        else:
            raise ValueError(
                f"`label_image` has dtype {label_image.dtype}, which is unable "
                "to be interpreted as an label."
            )
            
        if self.labels is not None and not new:
            if label_image.shape != self.labels.shape:
                raise ImageAxesError(
                    "Shape mismatch. Existing labels have shape "
                    f"{self.labels.shape} while labels with shape "
                    f"{label_image.shape} is given."
                )
            
            self.labels = self.labels.add_label(label_image)
        else:
            # when label_image is simple ndarray
            if not isinstance(label_image, MetaArray):
                if label_image.shape == self.shape:
                    axes = self.axes
                elif label_image.ndim == 2 and self.axes.contains("yx"):
                    axes = "yx"
                else:
                    raise ValueError("Could not infer axes of `label_image`.")
            else:
                axes = label_image.axes
                if not self.axes.contains(label_image.axes):
                    raise ImageAxesError(
                        f"Axes mismatch. Image has {self.axes}-axes but "
                        f"{axes} was given."
                    )
                
            self.labels = Label(label_image, axes=axes, source=self.source)
        return self.labels
    
    @check_input_and_output(need_labels=True)
    def proj_labels(self, axis=None, forbid_overlap=False) -> Label:
        """
        Label projection. This function is useful when zyx-labels are drawn but you want to reduce the 
        dimension.

        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        forbid_overlap : bool, default is False
            If True and there were any label overlap, this function will raise ValueError.

        Returns
        -------
        Label
            Projected labels.
        """
        self.labels = self.labels.proj(axis=axis, forbid_overlap=forbid_overlap)
        return self.labels
    
    def split(self, axis=None) -> DataList[Self]:
        """
        Split n-dimensional image into (n-1)-dimensional images. This function is different from
        `np.split`, which split an array into smaller pieces (n-D to n-D).

        Parameters
        ----------
        axis : str or int, optional
            Along which axis the original image will be split, by default "c"

        Returns
        -------
        list of arrays
            Separate images
        """
        # determine axis in int.
        if axis is None:
            axis = find_first_appeared(self.axes, include="cztpa")
        axisint = self.axisof(axis)
        
        imgs = super().split(axisint)
        if self.labels is not None:
            labels = self.labels.split(axisint)
            for img, lbl in zip(imgs, labels):
                lbl.axes = self.labels.axes.drop(axisint)
                lbl.set_scale(self.labels)
                img.labels = lbl
            
        return imgs
        
    def tile(
        self,
        shape: tuple[int, int] | None = None,
        along: str | None = None,
        order: str | None = None,
    ) -> Self:
        """
        Tile images in a certain order.

        Parameters
        ----------
        shape : tuple[int, int], optional
            Grid shape. This parameter must be specified unless the length of `along` is 2.
        along : str, optional
            Axis (Axes) over which will be iterated.
        order : str, {"r", "c"}, optional
            Order of iteration. "r" means row-wise and "c" means column-wise.
        
            row-wise
                ----->
                ----->
                ----->
            
            column-wise
                | | |
                | | |
                v v v

        Returns
        -------
        Labeled
            Tiled array
        """        
        if along is None:
            for a in self.axes:
                l = np.prod(shape)
                if self.sizeof(a) == l:
                    along = a
                    break
            else:
                raise ValueError(f"Could not find axis that can be reshaped to shape {shape}")
        elif len(along) == 2:
            uyaxis, uxaxis = self.axisof(along[0]), self.axisof(along[1])
            if uyaxis < uxaxis:
                shape = self.sizesof(along)
                order = "r"
            else:
                order = "c"
                shape = self.sizesof(along[::-1])
        elif len(along) == 1:
            if shape is None:
                raise ValueError("`shape` must be specified unless the length of `along` is 2.")
        else:
            raise ValueError("`along` must be a string with length 1 or 2.")
        
        if order is None:
            order = "r"
            
        uy_max, ux_max = shape
        imgy, imgx = self.sizesof("yx")
        if len(shape) == 2:
            c_axes = complement_axes("yx"+along, self.axes)
            new_axes = c_axes + "yx"
            outshape = self.sizesof(c_axes) + (uy_max*imgy, ux_max*imgx)
        else:
            raise ValueError("Shape mismatch")
        
        out = np.zeros(outshape, dtype=self.dtype)
        
        if order == "r":
            iter_tile = _iter_tile_yx
        elif order == "c":
            iter_tile = _iter_tile_xy
        else:
            raise ValueError(f"Could not interpret order={repr(order)}.")
        
        for (_, img), sl in zip(self.iter(along), iter_tile(uy_max, ux_max, imgy, imgx)):
            out[sl] = img
            
        out = out.view(self.__class__)
        out._set_info(self, new_axes=new_axes)
        
        if self.labels is not None:
            tiled_label = self.labels.tile(shape, along, order)
            out.labels = tiled_label
        return out
    
    @check_input_and_output
    def for_each_channel(self, func: str, along: str = "c", **kwargs) -> Self:
        """
        Apply same function with different parameters for each channel. This function will be useful
        when the parameters are dependent on channels, like wave length.

        Parameters
        ----------
        func : str
            Function name to apply over channel axis.
        along : str, default is "c"
            Along which axis function will be applied to.

        Returns
        -------
        LabeledArray
            output image stack
        """        
        if not hasattr(self, func):
            raise AttributeError(f"{self.__class__} does not have method {func}")
        imgs = self.split(along)
        outs = []
        for img, kw in zip(imgs, _iter_dict(kwargs, len(imgs))):
            outs.append(out)
        out = np.stack(outs, axis=along)
        return out
    
    @check_input_and_output
    def for_params(self, func: Callable|str, var: dict[str, Iterable] = None, **kwargs) -> DataList:
        """
        Apply same function with different parameters with same input. This function will be useful
        when you want to try different conditions to the same image.

        Parameters
        ----------
        func : callable or str
            Function to apply repetitively. If str, then member method will be called. 
        var : dict[str, Iterable], optional
            Name of variable and the values to try. If you want to try sigma=1,2,3 then you should
            give `var={"sigma": [1, 2, 3]}`.
        kwargs
            Fixed paramters that will be passed to `func`. If `var` is not given and only one parameter
            is provided in `kwargs`, then kwargs will be `var`.

        Returns
        -------
        DataList
            List of outputs.
            
        Example
        -------
        1. Try LoG filter with different Gaussian kernel size and visualize all of them in napari.
            
            >>> out = img.for_params("log_filter", var={"sigma":[1, 2, 3, 4]})
            # or
            >>> out = img.for_params("log_filter", sigma=[1, 2, 3, 4])
            # then
            >>> ip.gui.add(out)
        """        
        if isinstance(func, str) and hasattr(self, func):
            f = getattr(self, func)
        elif callable(func):
            f = partial(func, self)
        elif not callable(func):
            raise AttributeError(f"{func} is neither {self.__class__}'s' method nor callable object.")
        
        if isinstance(var, dict):
            key, values = tuple(var.items())[0]
        elif var is None and len(kwargs) == 1:
            key, values = tuple(kwargs.items())[0]
            kwargs = dict()
        else:
            raise ValueError("Wrong inputs.")
        
        if key in kwargs.keys():
            raise ValueError(f"Keyword {key} exists in `kwargs`.")
        outlist = DataList()
        
        for v in values:
            kwargs[key] = v
            out = f(**kwargs)
            outlist.append(out)
        return outlist        
        
    
    @check_input_and_output(need_labels=True)
    def extract(self, label_ids=None, filt=None, cval:float=0) -> DataList[Self]:
        """
        Extract certain regions of the image and substitute others to `cval`.

        Parameters
        ----------
        label_ids : int or iterable of int, by default all the label IDs.
            Which label regions are extracted.
        filt : callable, optional
            If given, only regions `X` that satisfy filt(self, X) will extracted.
        cval : float, default is 0.
            Constant value to fill regions outside the extracted labeled regions.
            
        Returns
        -------
        DataList of LabeledArray
            Extracted image(s)
        """        
        if filt is None:
            filt = lambda arr, lbl: True
        elif not callable(filt):
            raise TypeError("`filt` must be callable if given.")
        
        if np.isscalar(label_ids):
            label_ids = [label_ids]
        elif label_ids is None:
            # All the labels except for 0 (which means not labeled)
            label_ids = [i for i in np.unique(self.labels) if i != 0]
        
        slices = ndi.find_objects(self.labels)
        out = []
        for i in label_ids:
            sl = slices[i-1]
            obj = self[sl]
            subregion = obj.labels > 0
            if filt(self, subregion):
                obj.value[~subregion] = cval
                del obj.labels
                out.append(obj)
                
        out = DataList(out)
            
        return out

def _iter_dict(d, nparam):
    out = dict()
    for i in range(nparam):
        for k, v in d.items():
            if isinstance(v, list):
                if len(v) != nparam:
                    # raise error here for an earlier feedback.
                    raise ValueError(f"Number of parameter '{k}' does not match the number channels.")
                out[k] = v[i]
            else:
                out[k] = v
        yield out

class NotMe:
    def __eq__(self, other):
        return False

_notme = NotMe()

def _shape_match(img: LabeledArray, label: Label):
    """
    e.g.)
    img   ... 12(t), 100(y), 50(x)
    label ... 100(y), 50(x)
        -> True
    img   ... 12(t), 100(y), 50(x)
    label ... 30(y), 50(x)
        -> False
    """    
    img_shape = img.shape
    label_shape = label.shape
    return all(
        [getattr(img_shape, str(a), _notme) == getattr(label_shape, str(a), _notme)
         for a in label.axes]
    )

def _iter_tile_yx(ymax, xmax, imgy, imgx):
    """
    +--+--+--+
    | 0| 1| 2|
    +--+--+--+
    | 3| 4| 5|
    +--+--+--+
    | 6| 7|..|
    +--+--+--+
    """    
    for uy, ux in itertools.product(range(ymax), range(xmax)):
        sly = slice(uy*imgy, (uy+1)*imgy, None)
        slx = slice(ux*imgx, (ux+1)*imgx, None)
        yield ..., sly, slx

def _iter_tile_xy(ymax, xmax, imgy, imgx):
    """
    +--+--+--+
    | 0| 3| 6|
    +--+--+--+
    | 1| 4| 7|
    +--+--+--+
    | 2| 5|..|
    +--+--+--+
    """    
    for uy, ux in itertools.product(range(xmax), range(ymax)):
        sly = slice(uy*imgy, (uy+1)*imgy, None)
        slx = slice(ux*imgx, (ux+1)*imgx, None)
        yield ..., slx, sly