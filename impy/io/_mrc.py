from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
import warnings
import numpy as np
from impy.io._registry import IO
from impy.io._utils import rechunk_to_ones, MemmapArrayWriter, ImpyArray, ImageData, ImageMetadata

if TYPE_CHECKING:
    from mrcfile.mrcobject import MrcObject

@IO.mark_reader(".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz", ".mrcs")
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


@IO.mark_header_reader(".mrc", ".rec", ".st", ".map", ".mrc.gz", ".map.gz", ".mrcs")
def _(path: str) -> ImageData:
    """The MRC header reader"""
    import mrcfile

    with mrcfile.mmap(path, permissive=True, mode="r") as mrc:
        meta = _parse_mrcfile(mrc)

    return meta

@IO.mark_writer(".mrc", ".rec", ".st", ".map", ".mrcs")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The MRC writer."""

    import mrcfile

    input_axes = [str(a) for a in img.axes]

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
    elif len(voxel_size) > 3:
        voxel_size = voxel_size[:3]

    if img.dtype == "bool":
        img = img.astype(np.int8)
    elif img.dtype == "float64":
        warnings.warn(
            "MRC format does not support float64. Converting to float32.",
            UserWarning,
            stacklevel=2,
        )
        img = img.astype(np.float32)
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
    if Path(path).exists():
        ctx_mgr = mrcfile.open(path, mode="r+")
    else:
        ctx_mgr = mrcfile.new(path)
    with ctx_mgr as mrc:
        mrc.set_data(img.value)
        mrc.voxel_size = voxel_size
        if input_axes == ["t", "y", "x"]:
            mrc.set_image_stack()

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

    metadata = {}
    scale = dict.fromkeys(axes, 1.0)
    for a in axes:
        if a in "zyx":
            this_scale = mrc.voxel_size[a] / 10
            if this_scale == 0:
                this_scale = 1.0
            scale[a] = this_scale
    try:
        orig_x, orig_y, orig_z = mrc.header["origin"].item()
    except Exception:
        pass
    else:
        metadata["origin_nm"] = np.array([orig_z, orig_y, orig_x], dtype=np.float32) / 10

    metadata["labels"] = mrc.get_labels()

    return ImageMetadata(
        axes=axes,
        scale=scale,
        unit={a: "nm" for a in axes},
        metadata=metadata,
        labels=None,
    )
