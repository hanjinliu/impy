from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings
import os
import numpy as np
from impy.io._registry import IO
from impy.io._utils import rechunk_to_ones, MemmapArrayWriter, ImpyArray, ImageData, ImageMetadata

from impy.axes import ImageAxesError

if TYPE_CHECKING:
    from mrcfile.mrcobject import MrcObject

@IO.mark_reader(".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz")
def _(path: str, memmap: bool = False) -> ImageData:
    """The MRC format reader"""
    import mrcfile

    if memmap:
        open_func = mrcfile.mmap
    else:
        open_func = mrcfile.open

    with open_func(path, permissive=True, mode="r") as mrc:
        meta = _parse_mrcfile(mrc)
        image = mrc.data

    return ImageData.from_metadata(image, meta)


@IO.mark_header_reader(".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz")
def _(path: str) -> ImageData:
    """The MRC header reader"""
    import mrcfile

    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        meta = _parse_mrcfile(mrc)

    return meta

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
    if len(voxel_size) == 2:
        voxel_size = (1.0,) + voxel_size

    if img.dtype == "bool":
        img = img.astype(np.int8)

    if lazy:
        from dask import array as da

        mode = _MRC_MODE[img.dtype]
        mrc_mmap = mrcfile.new_mmap(path, img.shape, mrc_mode=mode, overwrite=True)
        mrc_mmap.voxel_size = voxel_size

        img_dask = rechunk_to_ones(img.value)
        writer = MemmapArrayWriter(path, mrc_mmap.data.offset, img.shape, img_dask.chunksize)
        da.store(img_dask, writer)
        return None

    # get voxel_size
    if img.axes not in ("zyx", "yx"):
        raise ImageAxesError(
            f"Can only save zyx- or yx- image as a mrc file, but image has {img.axes} "
            "axes."
        )
    if Path(path).exists():
        with mrcfile.open(path, mode="r+") as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = voxel_size

    else:
        with mrcfile.new(path) as mrc:
            mrc.set_data(img.value)
            mrc.voxel_size = voxel_size
    return None

_MRC_MODE = {
    np.dtype("int8"): 0,
    np.dtype("int16"): 1,
    np.dtype("float32"): 2,
    np.dtype("complex64"): 4,
    np.dtype("uint16"): 6,
    np.dtype("float16"): 12,
}

def _parse_mrcfile(mrc: MrcObject) -> ImageMetadata:
    if mrc.is_single_image():
        axes = "yx"
    elif mrc.is_volume_stack():
        axes = "tzyx"
    else:
        axes = "zyx"

    scale = dict.fromkeys(axes, 1.0)
    for a in axes:
        scale[a] = mrc.voxel_size[a] / 10

    return ImageMetadata(
        axes=axes,
        scale=scale,
        unit={a: "nm" for a in axes},
        metadata={},
        labels=None,
    )
