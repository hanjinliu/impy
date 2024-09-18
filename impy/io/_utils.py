from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union, NamedTuple, Protocol
import numpy as np

if TYPE_CHECKING:
    from dask.array.core import Array
    from impy.arrays.bases import MetaArray
    from impy.arrays import LazyImgArray
    from impy.axes import Axes
    from numpy.typing import DTypeLike

    ImpyArray = Union[MetaArray, LazyImgArray]
else:
    ImpyArray = Any

def rechunk_to_ones(arr: Array):
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

class MemmapArrayWriter:
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

    @classmethod
    def from_metadata(self, image: _ImageType, meta: ImageMetadata):
        return ImageData(
            image=image,
            axes=meta.axes,
            scale=meta.scale,
            unit=meta.unit,
            metadata=meta.metadata,
            labels=meta.labels,
        )

class ImageMetadata(NamedTuple):
    """Tuple of image info except for the image data itself."""

    axes: str | None
    scale: dict[str, float] | None
    unit: dict[str, str | None] | str | None
    metadata: dict[str, Any]
    labels: list[Any]  # channel labels

def get_channel_labels(axes: Axes):
    if "c" in axes:
        axis = axes["c"]
        if axis.labels is not None:
            return axis.labels
    return None
