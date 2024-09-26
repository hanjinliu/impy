from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence, TypeVar, Union
from pathlib import Path
import re
import os
import numpy as np
from impy.io._utils import ImageData

__all__ = [
    "imread",
    "imread_dask",
    "imsave",
    "mark_reader",
    "mark_writer",
]

if TYPE_CHECKING:
    from impy.arrays.bases import MetaArray
    from impy.arrays import LazyImgArray

    ImpyArray = Union[MetaArray, LazyImgArray]
    Reader = Callable[[str, bool], ImageData]
    _R = TypeVar("_R", bound=Reader)
    Writer = Callable[[str, ImpyArray, bool], None]
    _W = TypeVar("_W", bound=Writer)
    HeaderReader = Callable[[str], dict[str, Any]]
    _HR = TypeVar("_HR", bound=HeaderReader)
    HeaderWriter = Callable[[str, dict[str, Any]], None]
    _HW = TypeVar("_HW", bound=HeaderWriter)

    _T = TypeVar("_T")

class ImageIO:
    """A I/O class for image data."""

    def __init__(self):
        self._reader: dict[str, Reader] = {}
        self._default_reader: Reader | None = None
        self._writer: dict[str, Writer] = {}
        self._default_writer: Writer | None = None
        self._header_reader: dict[str, HeaderReader] = {}
        self._header_writer: dict[str, HeaderWriter] = {}

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

    def mark_header_reader(self, *ext: str) -> Callable[[_HR], _HR]:
        """
        Mark a function as a header reader function.

        Examples
        --------
        >>> @IO.mark_header_reader(".tif")
        >>> def read_header_tif(path):
        >>>     ...
        """
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: _HR):
            nonlocal ext_list
            for ext in ext_list:
                self._header_reader[ext] = f
            return f

        return _register

    def mark_header_writer(self, *ext: str) -> Callable[[_HW], _HW]:
        """
        Mark a function as a header writer function.

        Examples
        --------
        >>> @IO.mark_header_writer(".tif")
        >>> def write_header_tif(path, header):
        >>>     ...
        """
        ext_list: list[str] = []
        for _ext in ext:
            if not _ext.startswith("."):
                _ext = "." + _ext
            ext_list.append(_ext)

        def _register(f: _HW):
            nonlocal ext_list
            for ext in ext_list:
                self._header_writer[ext] = f
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
        reader = _get_ext(self._reader, path) or self._default_reader
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
        from impy.array_api import xp

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
        writer = _get_ext(self._writer, path) or self._default_writer
        return writer(path, img, lazy)

    def read_header(self, path: str | Path) -> dict[str, Any]:
        """
        Read header of an image file.

        The reader is chosen according to the file extension.

        Parameters
        ----------
        path : str
            File path of an image.

        Returns
        -------
        dict
            Image header.
        """
        reader = _get_ext(self._header_reader, path)
        if reader is None:
            raise ValueError(f"Reader for {path} is not found.")
        return reader(str(path))

    def write_header(self, path: str | Path, header: dict[str, Any]) -> None:
        """
        Write header of an image file.

        The writer is chosen according to the file extension.

        Parameters
        ----------
        path : str
            File path of an image.
        header : dict
            Image header.
        """
        writer = _get_ext(self._header_writer, path)
        if writer is None:
            raise ValueError(f"Writer for {path} is not found.")
        writer(str(path), header)


def _chunk_to_slice(chunk: Sequence[int]) -> list[slice]:
    # _chunk_to_slice([5, 15, 30]) --> [0:5, 5:20, 20:50]
    start = 0
    out: list[slice] = []
    for c in chunk:
        _next = start + c
        out.append(slice(start, _next))
        start = _next
    return out

def _get_ext(reg: dict[str, _T], path: str) -> _T | None:
    sufs = Path(path).suffixes
    if sufs:
        for i in range(len(sufs)):
            ext = "".join(sufs[i:])
            if ext in reg:
                return reg[ext]
        return None
    else:
        return reg.get("", None)

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

@IO.mark_writer("")
def _(path: str, img: ImpyArray, lazy: bool = False):
    raise ValueError(f"Input path {path!r} does not have file extension.")

imread = IO.imread
imread_dask = IO.imread_dask
imsave = IO.imsave
mark_reader = IO.mark_reader
mark_writer = IO.mark_writer
read_header = IO.read_header
