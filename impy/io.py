from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple, Callable, Sequence, TypeVar, Union, Protocol
from pathlib import Path
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
        """Should return the number of dimensions of the image."""

    @property
    def dtype(self) -> np.dtype:
        """Should return the data type of the image."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Should return the shape of the image."""

    def astype(self, dtype: DTypeLike):
        """Should return an image with the specified data type."""


class ImageData(NamedTuple):
    """Tuple of image info."""

    image: _ImageType
    axes: str | None
    scale: dict[str, float] | None
    unit: dict[str, str | None] | str | None
    metadata: dict[str, Any]
    labels: list[Any]  # channel labels

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile
    from .arrays.bases import MetaArray
    from .arrays import LazyImgArray
    from .axes import Axes
    from dask.array.core import Array

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

    def imread(self, path: str | Path, memmap: bool = False) -> ImageData:
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
        ext = Path(path).suffix
        reader = self._reader.get(ext, self._default_reader)
        return reader(str(path), memmap)

    def _imread_slice(self, path, sl: tuple[slice, ...]) -> np.memmap:
        mmap = self.imread(path, memmap=True).image
        return np.asarray(mmap[sl], dtype=mmap.dtype)

    def imread_dask(self, path: str | Path, chunks: Any) -> ImageData:
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

        path = Path(path)
        image_data = self.imread(path, memmap=True)
        img = image_data.image

        from dask import array as da, delayed
        from dask.array.core import normalize_chunks

        if path.suffix == ".zarr":
            if img.dtype == ">u2":
                img = img.astype(np.uint16)
            dask = da.from_zarr(img, chunks=chunks).map_blocks(
                xp.asarray, dtype=img.dtype
            )
        else:
            chunks_: tuple[tuple[int, ...]] = normalize_chunks(
                chunks,
                shape=img.shape,
                dtype=img.dtype,
            )
            chunk_slices = [_chunk_to_slice(c) for c in chunks_]
            block_shape = tuple(len(c) for c in chunks_)
            delayed_imread = delayed(self._imread_slice)
            arr_blocks = np.empty(block_shape, dtype=object)
            for ind, _ in np.ndenumerate(arr_blocks):
                sl = tuple(sls[i] for i, sls in zip(ind, chunk_slices))
                cur_shape = tuple(_sl.stop - _sl.start for _sl in sl)
                arr_blocks[ind] = da.from_delayed(
                    delayed_imread(path, sl), shape=cur_shape, dtype=img.dtype,
                    meta=xp.array([]),
                )
            dask = da.block(arr_blocks.tolist())

        return ImageData(
            image=dask,
            axes=image_data.axes,
            scale=image_data.scale,
            unit=image_data.unit,
            metadata=image_data.metadata,
            labels=image_data.labels,
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

def _chunk_to_slice(chunk: Sequence[int]) -> list[slice]:
    # _chunk_to_slice([5, 15, 30]) --> [0:5, 5:20, 20:50]
    start = 0
    out: list[slice] = []
    for c in chunk:
        _next = start + c
        out.append(slice(start, _next))
        start = _next
    return out

IO = ImageIO()


@IO.set_default_reader
def _(path: str, memmap: bool = False):
    """By default use skimage reader."""
    from skimage import io
    img: np.ndarray = io.imread(path)
    _, ext = os.path.splitext(path)
    labels = None
    if ext in (".png", ".jpg") and img.ndim == 3 and img.shape[-1] <= 4:
        axes = "yxc"
        if img.shape[-1] == 3:
            labels = ["R", "G", "B"]
        elif img.shape[-1] == 4:
            labels = ["R", "G", "B", "A"]
    elif img.ndim == 2:
        axes = "yx"
    else:
        axes = None
    return ImageData(
        image=img,
        axes=axes,
        scale=None,
        unit=None,
        metadata={},
        labels=labels,
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
    from tifffile import TiffFile, xml2dict
    with TiffFile(path) as tif:
        ijmeta = tif.imagej_metadata
        series0 = tif.series[0]

        pagetag = series0.pages[0].tags

        if ijmeta is None:
            ijmeta = {}

        ijmeta.pop("ROI", None)

        try:
            axes = series0.axes.lower()
        except Exception:
            axes = None

        tags = {v.name: v.value for v in pagetag.values()}

        labels = ijmeta.get("Labels")
        scale = dict()
        dz = ijmeta.get("spacing", 1.0)
        scale_unit: dict[str, float] = {}
        try:
            # For MicroManager
            info = _load_json(ijmeta["Info"])
            ijmeta["Info"] = info  # update string to dict
            # try to read time scale
            if "t" in axes:
                dt = info.get("Interval_ms", None)
                if dt is not None and dt > 0:
                    scale["t"] = dt
                    scale_unit["t"] = "ms"
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
        _unit = ijmeta.pop("unit", None)
        for k in scale.keys():
            scale_unit[k] = _unit
        if tif.is_ome:
            try:
                xml = xml2dict(tif.ome_metadata)
                meta_data: dict[str, Any] = xml["OME"]["Image"]
                if isinstance(meta_data, list):
                    meta_data = meta_data[0]
                pix_info: dict[str, Any] = meta_data.get("Pixels", {})
                # get scale
                scale_unit = {x.lower(): pix_info.get(f"PhysicalSize{x}Unit") for x in "ZYX"}
                dz, dy, dx = (pix_info.get(f"PhysicalSize{x}", 1.0) for x in "ZYX")
                scale["z"] = dz
                scale["y"] = dy
                scale["x"] = dx
                # get channel names
                chn = pix_info.get("Channel")
                if isinstance(chn, (list, tuple)):
                    labels = [ch.get("Name") for ch in chn]
                else:
                    labels = None
                # get time interval

            except Exception:
                pass

        if memmap:
            image = tif.asarray(out="memmap")
        else:
            image = tif.asarray()


    return ImageData(
        image=image,
        axes=axes,
        scale=scale,
        unit=scale_unit,
        metadata=ijmeta,
        labels=labels,
    )

@IO.mark_writer(".tif", ".tiff")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The TIFF writer."""
    if lazy:
        from tifffile import memmap
        from dask import array as da
        kwargs = _get_ijmeta_from_img(img, update_lut=False)
        mmap = memmap(str(path), shape=img.shape, dtype=img.dtype, **kwargs)
        img_dask = _rechunk_to_ones(img.value)
        writer = _MemmapArrayWriter(path, mmap.offset, img.shape, img_dask.chunksize)
        da.store(img_dask, writer)
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

        warnings.warn("Image axes changed", UserWarning, stacklevel=2)

    img = img.sort_axes()
    if img.dtype == "bool":
        img = img.astype(np.uint8)
    imsave_kwargs = _get_ijmeta_from_img(img, update_lut=True)
    imwrite(path, img, **imsave_kwargs)
    return None

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

    with open_func(path, permissive=True, mode="r") as mrc:
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
        unit={a: "nm" for a in axes},
        metadata={},
        labels=None,
    )


@IO.mark_writer(".mrc", ".rec", ".st", ".map")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The MRC writer."""

    import mrcfile

    if img.scale_unit == "nm":
        voxel_size = tuple(np.array(img.scale)[::-1] * 10)
    elif img.scale_unit in ("ang", "Å", "angstrom"):
        voxel_size = tuple(np.array(img.scale)[::-1])
    else:
        warnings.warn(
            f"Scale unit was {img.scale_unit}. Could not normalize scale."
        )
        voxel_size = (1.0, 1.0, 1.0)

    if img.dtype == "bool":
        img = img.astype(np.int8)

    if lazy:
        from dask import array as da

        mode = _MRC_MODE[img.dtype]
        mrc_mmap = mrcfile.new_mmap(path, img.shape, mrc_mode=mode, overwrite=True)
        mrc_mmap.voxel_size = voxel_size

        img_dask = _rechunk_to_ones(img.value)
        writer = _MemmapArrayWriter(path, mrc_mmap.data.offset, img.shape, img_dask.chunksize)
        da.store(img_dask, writer)
        return None

    # get voxel_size
    if img.axes not in ("zyx", "yx"):
        raise ImageAxesError(
            f"Can only save zyx- or yx- image as a mrc file, but image has {img.axes} axes."
            )
    if os.path.exists(path):
        with mrcfile.open(path, mode="r+") as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = voxel_size

    else:
        with mrcfile.new(path) as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = voxel_size
    return None

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
        unit=zf.attrs.get("unit", None),
        metadata=zf.attrs.get("metadata", {}),
        labels=zf.attrs.get("labels", None),
    )


@IO.mark_writer(".zarr")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The zarr writer."""
    import zarr

    f = zarr.open(path, mode="w")
    metadata = img.metadata.copy()

    f.attrs["axes"] = str(img.axes)
    f.attrs["scale"] = {str(a): v for a, v in img.scale.items()}
    f.attrs["metadata"] = metadata
    f.attrs["unit"] = img.scale_unit
    f.attrs["labels"] = _get_channel_labels(img.axes)
    if lazy:
        img.value.to_zarr(url=os.path.join(path, "data"))
    else:
        z = img.value
        f.array("data", z)
    return None


@IO.mark_reader(".npy")
def _(path: str, memmap: bool = False) -> ImageData:
    """The numpy reader."""
    if memmap:
        image = np.load(path, mmap_mode="r", allow_pickle=True)
    else:
        image = np.load(path, allow_pickle=True)
    return ImageData(
        image=image,
        axes=None,
        scale=None,
        unit=None,
        metadata={},
        labels=None,
    )


@IO.mark_writer(".npy")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The numpy writer."""
    if lazy:
        raise NotImplementedError("Lazy saving is not implemented for npy files.")
    else:
        np.save(path, img.value)
    return None


@IO.mark_reader(".npz")
def _(path: str, memmap: bool = False) -> ImageData:
    """The numpy reader."""
    npz: NpzFile = np.load(path, mmap_mode="r", allow_pickle=True)

    none_scalar = _scalar(None)
    metadata = npz.get("metadata", none_scalar).item()
    if metadata is None:
        metadata = {}
    return ImageData(
        image=npz["data"],
        axes=npz.get("axes", none_scalar).item(),
        scale=npz.get("scale", none_scalar).item(),
        unit=npz.get("unit", none_scalar).item(),
        metadata=metadata,
        labels=npz.get("labels", none_scalar).item(),
    )


@IO.mark_writer(".npz")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The numpy writer."""
    if lazy:
        raise NotImplementedError("Lazy saving is not implemented for npz files.")
    else:
        np.savez(
            path,
            data=img.value,
            axes=_scalar(str(img.axes)),
            scale=_scalar(img.scale.asdict()),
            metadata=_scalar(img.metadata),
            unit=_scalar(img.scale_unit),
            labels=_scalar(_get_channel_labels(img.axes)),
        )
    return None

@IO.mark_reader(".nd2")
def _(path: str, memmap: bool = False):
    import nd2
    from dataclasses import is_dataclass, asdict

    with nd2.ND2File(path) as f:
        if not memmap:
            image = f.asarray()
        else:
            image = f.to_dask()
        axes = "".join(f.sizes.keys()).lower()
        vsize = f.voxel_size()
        scale = dict(z=vsize.z, y=vsize.y, x=vsize.x)
        metadata = f.metadata
        labels = None
        if "c" in axes:
            try:
                labels = []
                for chn in metadata.channels:
                    labels.append(chn.channel.name)
            except Exception:
                pass
        if is_dataclass(metadata):
            metadata = asdict(metadata)

    return ImageData(
        image=image,
        axes=axes,
        scale=scale,
        unit="µm",
        metadata=metadata,
        labels=labels,
    )

@IO.mark_writer("")
def _(path: str, img: ImpyArray, lazy: bool = False):
    raise ValueError(f"Input path {path!r} does not have file extension.")

_MRC_MODE = {
    np.dtype("int8"): 0,
    np.dtype("int16"): 1,
    np.dtype("float32"): 2,
    np.dtype("complex64"): 4,
    np.dtype("uint16"): 6,
    np.dtype("float16"): 12,
}

def _load_json(s: str) -> dict[str, Any]:
    return json.loads(re.sub("'", '"', s))

def _get_ijmeta_from_img(img: MetaArray, update_lut=True):
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
    except Exception:
        info = {}
    metadata["Info"] = str(info)
    scale_unit = img.scale_unit
    if scale_unit[0] == "μ":
        scale_unit = "\\u00B5" + scale_unit[1:]
    metadata["unit"] = scale_unit

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

def _get_channel_labels(axes: Axes):
    if "c" in axes:
        axis = axes["c"]
        if axis.labels is not None:
            return axis.labels
    return None

def _scalar(x: Any) -> np.ndarray:
    ar = np.array(None, dtype=object)
    ar[()] = x
    return ar

def _rechunk_to_ones(arr: Array):
    """Rechunk the array to (1, 1, ..., 1, n, Ny, Nx)"""
    size = np.prod(arr.chunksize)
    shape = arr.shape
    cur_prod = 1
    max_i = arr.ndim
    for i in reversed(range(arr.ndim)):
        cur_prod *= shape[i]
        if cur_prod > size:
            break
        max_i = i
    nslices = max(int(size / np.prod(shape[max_i:])), 1)
    if max_i == 0:
        return arr
    else:
        return arr.rechunk((1,) * (max_i - 1) + (nslices,) + shape[max_i:])

class _MemmapArrayWriter:
    def __init__(
        self,
        path: str,
        offset: int,
        shape: tuple[int, ...],
        chunksize: tuple[int, ...],
    ):
        self._path = path
        self._offset = offset
        self._shape = shape  # original shape
        self._chunksize = chunksize  # chunk size
        # shape = (33, 160, 1000, 1000)
        # chunksize = (1, 16, 1000, 1000)
        border = 0
        for i, c in enumerate(chunksize):
            if c != 1:
                border = i
                break
        self._border = border

    def __setitem__(self, sl: tuple[slice, ...], arr: np.ndarray):
        # efficient: shape = (10, 100, 150) and sl = (3:5, 0:100, 0:150)
        # sl = (0:1, 16:32, 0:1000, 0:1000)

        offset = np.sum([sl[i].start * arr.strides[i] for i in range(self._border + 1)])
        arr_ravel = arr.ravel()
        mmap = np.memmap(
            self._path,
            mode="r+",
            offset=self._offset + offset,
            shape=arr_ravel.shape,
            dtype=arr.dtype,
        )
        mmap[:arr_ravel.size] = arr_ravel
        mmap.flush()
