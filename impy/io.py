from __future__ import annotations
from typing import TYPE_CHECKING, Any, NamedTuple, Callable, TypeVar, Union, Protocol
import json
import re
import warnings
import os
import numpy as np
from numpy.typing import DTypeLike

from .axes import ImageAxesError
from .utils.axesop import complement_axes

__all__ = [
    "imread",
    "imread_dask",
    "imsave",
    "mark_reader",
    "mark_writer",
]

class _ImageType(Protocol):
    @property
    def ndim(self) -> int:
        ...
    
    @property
    def dtype(self) -> np.dtype:
        ...
    
    def astype(self, dtype: DTypeLike):
        ...


class ImageData(NamedTuple):
    """Tuple of image info."""
    
    image: _ImageType
    axes: str | None
    scale: dict[str, float] | None
    metadata: dict[str, Any]


if TYPE_CHECKING:
    from .arrays.bases import MetaArray
    from .arrays import LazyImgArray
    ImpyArray = Union[MetaArray, LazyImgArray]
    Reader = Callable[[str, bool], ImageData]
    _R = TypeVar("_R", bound=Reader)
    Writer = Callable[[str, ImpyArray, bool], None]
    _W = TypeVar("_W", bound=Writer)


class ImageIO:
    """A I/O class for image data."""
    
    def __init__(self):
        self._reader: dict[str, Reader] = {}
        self._default_reader: Reader | None = None
        self._writer: dict[str, Writer] = {}
        self._default_writer: Writer | None = None
    
    def set_default_reader(self, f: _R) -> _R:
        self._default_reader = f
        return f
    
    def set_default_writer(self, f: _W) -> _W:
        self._default_writer = f
        return f

    def mark_reader(self, *ext: str) -> Callable[[_R], _R]:
        """
        Mark a function as a reader function.
        
        Examples
        --------
        >>> @IO.mark_reader(".tif")
        >>> def read_tif(path, memmap=False):
        >>>     ...
        """
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: _R):
            nonlocal ext_list
            for ext in ext_list:
                self._reader[ext] = f
            return f

        return _register

    def mark_writer(self, *ext: str) -> Callable[[_R], _R]:
        """
        Mark a function as a writer function.

        Examples
        --------
        >>> @IO.mark_writer(".tif")
        >>> def save_tif(path, img, lazy=False):
        >>>     ...
        """        
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: _R):
            nonlocal ext_list
            for ext in ext_list:
                self._writer[ext] = f
            return f

        return _register
    
    def imread(self, path: str, memmap: bool = False) -> ImageData:
        """
        Read an image file.
        
        The reader is chosen according to the file extension.

        Parameters
        ----------
        path : str
            File path of an image.
        memmap : bool, default is False
            Read image as a memory-mapped-like state.

        Returns
        -------
        ImageData
            Image data tuple.
        """
        _, ext = os.path.splitext(path)
        reader = self._reader.get(ext, self._default_reader)
        return reader(path, memmap)
    
    def imread_dask(self, path: str, chunks: Any) -> ImageData:
        """
        Read an image file as a dask array.

        The reader is chosen according to the file extension.
        
        Parameters
        ----------
        path : str
            File path of an image.
        chunks : Any
            Parameter that will be passed to ``dask.array.from_array`` or
            ``dask.array.from_zarr`` function.

        Returns
        -------
        ImageData
            Image data tuple.
        """
        from .array_api import xp
        image_data = self.imread(path, memmap=True)
        img = image_data.image
        
        if img.dtype == ">u2":
            img = img.astype(np.uint16)
        
        from dask import array as da
        if str(type(img)) == "<class 'zarr.core.Array'>":
            dask = da.from_zarr(img, chunks=chunks).map_blocks(
                xp.asarray, dtype=img.dtype
            )
        else:
            dask = da.from_array(img, chunks=chunks, meta=xp.array([])).map_blocks(
                xp.asarray, dtype=img.dtype
            )
        return ImageData(
            image=dask, 
            axes=image_data.axes,
            scale=image_data.scale,
            metadata=image_data.metadata,
        )
    
    def imsave(
        self,
        path: str,
        img: ImpyArray,
        lazy: bool = False
    ) -> None:
        _, ext = os.path.splitext(path)
        writer = self._writer.get(ext, self._default_writer)
        return writer(path, img, lazy)


IO = ImageIO()


@IO.set_default_reader
def _(path: str, memmap: bool = False):
    """By default use skimage reader."""
    from skimage import io
    img: np.ndarray = io.imread(path)
    _, ext = os.path.splitext(path)
    if ext in (".png", ".jpg") and img.ndim == 3 and img.shape[-1] <= 4:
        axes = "yxc"
    elif img.ndim == 2:
        axes = "yx"
    else:
        axes = None
    return ImageData(
        image=img,
        axes=axes,
        scale=None,
        metadata={},
    )

@IO.set_default_writer
def _(path: str, img: ImpyArray, lazy: bool = False):
    """By default use skimage writer."""
    from skimage import io
    io.imsave(path, np.asarray(img.value), check_contrast=False)
    return None

@IO.mark_reader(".tif", ".tiff")
def _(path: str, memmap: bool = False) -> ImageData:
    """The tif file reader."""
    from tifffile import TiffFile
    with TiffFile(path) as tif:
        ijmeta = tif.imagej_metadata
        series0 = tif.series[0]
    
        pagetag = series0.pages[0].tags
        
        if ijmeta is None:
            ijmeta = {}
        
        ijmeta.pop("ROI", None)
        
        try:
            axes = series0.axes.lower()
        except:
            axes = None
        
        tags = {v.name: v.value for v in pagetag.values()}
        
        if memmap:
            image = tif.asarray(out="memmap")
        else:
            image = tif.asarray()
    
    # get scale
    scale = dict()
    dz = ijmeta.get("spacing", 1.0)
    try:
        # For MicroManager
        info = _load_json(ijmeta["Info"])
        dx = dy = info["PixelSize_um"]
    except Exception:
        try:
            xres = tags["XResolution"]
            yres = tags["YResolution"]
            dx = xres[1]/xres[0]
            dy = yres[1]/yres[0]
        except KeyError:
            dx = dy = dz
    
    scale["x"] = dx
    scale["y"] = dy
    # read z scale if needed
    if axes is not None and "z" in axes:
        scale["z"] = dz

    return ImageData(
        image=image,
        axes=axes,
        scale=scale,
        metadata=ijmeta,
    )

@IO.mark_reader(".mrc", ".rec", ".st", ".map", ".gz")
def _(path: str, memmap: bool = False) -> ImageData:
    """The MRC format reader"""
    import mrcfile
    if path.endswith(".gz") and not path.endswith(".map.gz"):
        raise ValueError("Only .map.gz file is supported.")

    if memmap:
        open_func = mrcfile.mmap
    else:
        open_func = mrcfile.open
    
    with open_func(path, mode="r") as mrc:
        metadata = {"unit": "nm"}
        ndim = len(mrc.voxel_size.item())
        if ndim == 3:
            axes = "zyx"
        elif ndim == 2:
            axes = "yx"
        else:
            raise RuntimeError(f"ndim = {ndim} not supported")
        
        scale = dict.fromkeys(axes, 1.0)
        for a in axes:
            scale[a] = mrc.voxel_size[a] / 10
        
        image = mrc.data
    
    return ImageData(
        image=image,
        axes=axes,
        scale=scale,
        metadata=metadata,
    )


@IO.mark_reader(".zarr")
def _(path: str, memmap: bool = False) -> ImageData:
    """The zarr reader."""
    import zarr
    zf = zarr.open(path, mode="r")
    if memmap:
        image = zf["data"]
    else:
        image = np.asarray(zf["data"])
        
    return ImageData(
        image=image,
        axes=zf.attrs.get("axes", None),
        scale=zf.attrs.get("scale", None),
        metadata=zf.attrs.get("metadata", {})
    )


@IO.mark_writer(".tif", ".tiff")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The TIFF writer."""
    if lazy:
        from tifffile import memmap
        kwargs = _get_ijmeta_from_img(img, update_lut=False)
        mmap = memmap(str(path), shape=img.shape, dtype=img.dtype, **kwargs)
        mmap[:] = img.value
        mmap.flush()
        return
    
    from tifffile import imwrite
    rest_axes = complement_axes(img.axes, "tzcyx")
    new_axes = ""
    for a in img.axes:
        if a in ["t", "z", "c", "y", "x"]:
            new_axes += a
        else:
            if len(rest_axes) == 0:
                raise ImageAxesError(f"Cannot save image with axes {img.axes}")
            new_axes += rest_axes[0]
            rest_axes = rest_axes[1:]
    
    # make a copy of the image for saving
    if new_axes != img.axes:
        img_new = img.copy()
        img_new.axes = new_axes
        img_new.set_scale(img)
        img = img_new
        
        warnings.warn("Image axes changed", UserWarning)
    
    img = img.sort_axes()
    imsave_kwargs = _get_ijmeta_from_img(img, update_lut=True)
    imwrite(path, img, **imsave_kwargs)
    return None

@IO.mark_writer(".mrc", ".rec", ".st", ".map")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The MRC writer."""
    if img.scale_unit and img.scale_unit != "nm":
        raise ValueError(
            f"Scale unit {img.scale_unit} is not supported. Convert to nm instead."
        )
    
    import mrcfile
    
    if lazy:
        mode = _MRC_MODE[img.dtype]
        mrc_mmap = mrcfile.new_mmap(path, img.shape, mrc_mode=mode, overwrite=True)
        mrc_mmap.voxel_size = tuple(np.array(img.scale)[::-1] * 10)
        mrc_mmap.data[:] = img.value
        mrc_mmap.flush()
        return None

    # get voxel_size
    if img.axes not in ("zyx", "yx"):
        raise ImageAxesError(
            f"Can only save zyx- or yx- image as a mrc file, but image has {img.axes} axes."
            )
    if os.path.exists(path):
        with mrcfile.open(path, mode="r+") as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = tuple(np.array(img.scale)[::-1] * 10)
            
    else:
        with mrcfile.new(path) as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = tuple(np.array(img.scale)[::-1] * 10)
    return None

@IO.mark_writer(".zarr")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The zarr writer."""
    import zarr
    f = zarr.open(path, mode="w")
    metadata = img.metadata.copy()
    metadata["unit"] = img.scale_unit
    
    f.attrs["axes"] = str(img.axes)
    f.attrs["scale"] = {str(a): v for a, v in img.scale.items()}
    f.attrs["metadata"] = metadata
    if lazy:
        img.value.to_zarr(url=os.path.join(path, "data"))
    else:
        z = img.value
        f.array("data", z)
    return None


_MRC_MODE = {
    np.int8: 0,
    np.int16: 1,
    np.float32: 2,
    np.complex64: 4,
    np.uint16: 6,
    np.float16: 12,
}

def _load_json(s: str):
    return json.loads(re.sub("'", '"', s))

def _get_ijmeta_from_img(img: "MetaArray", update_lut=True):
    metadata = img.metadata.copy()
    scale_view = img.scale
    if update_lut:
        lut_min, lut_max = np.percentile(img, [1, 99])
        metadata.update({"min": lut_min, "max": lut_max})
    # set lateral scale
    try:
        res = (1/scale_view["x"], 1/scale_view["y"])
    except Exception:
        res = None
    # set z-scale
    if "z" in img.axes:
        metadata["spacing"] = scale_view["z"]
    else:
        metadata["spacing"] = scale_view["x"]
        
    try:
        info = _load_json(metadata["Info"])
    except:
        info = {}
    metadata["Info"] = str(info)
    metadata["unit"] = img.scale_unit
        
    # set axes in tiff metadata
    metadata["axes"] = str(img.axes).upper()
    if img.ndim > 3:
        metadata["hyperstack"] = True
    
    return dict(imagej=True, resolution=res, metadata=metadata)

imread = IO.imread
imread_dask = IO.imread_dask
imsave = IO.imsave
mark_reader = IO.mark_reader
mark_writer = IO.mark_writer