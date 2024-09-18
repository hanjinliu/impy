from __future__ import annotations

from typing import Any, TYPE_CHECKING
import numpy as np
from impy.io._registry import IO
from impy.io._utils import get_channel_labels, ImpyArray, ImageData

if TYPE_CHECKING:
    from numpy.lib.npyio import NpzFile

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
            labels=_scalar(get_channel_labels(img.axes)),
        )
    return None

def _scalar(x: Any) -> np.ndarray:
    ar = np.array(None, dtype=object)
    ar[()] = x
    return ar
