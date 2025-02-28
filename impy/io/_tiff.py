from __future__ import annotations

from typing import Any, TYPE_CHECKING
import warnings
import json
import re
import numpy as np
from impy.io._registry import IO
from impy.io._utils import rechunk_to_ones, MemmapArrayWriter, ImpyArray, ImageData, ImageMetadata

from impy.axes import ImageAxesError
from impy.utils.axesop import complement_axes

if TYPE_CHECKING:
    from tifffile import TiffFile

@IO.mark_reader(".tif", ".tiff", ".lsm")
def _(path: str, memmap: bool = False) -> ImageData:
    """The tif file reader."""
    from tifffile import TiffFile

    with TiffFile(path) as tif:
        meta = _parse_tifffile(tif)
        if memmap:
            image = tif.asarray(out="memmap")
        else:
            image = tif.asarray()

    return ImageData.from_metadata(image, meta)

@IO.mark_header_reader(".tif", ".tiff", ".lsm")
def _(path: str) -> ImageMetadata:
    """The tif header reader."""
    from tifffile import TiffFile

    with TiffFile(path) as tif:
        meta = _parse_tifffile(tif)

    return meta


@IO.mark_writer(".tif", ".tiff", ".lsm")
def _(path: str, img: ImpyArray, lazy: bool = False):
    """The TIFF writer."""
    from tifffile import imwrite, memmap

    if lazy:
        from dask import array as da
        kwargs = _get_ijmeta_from_img(img, update_lut=False)
        mmap = memmap(str(path), shape=img.shape, dtype=img.dtype, **kwargs)
        img_dask = rechunk_to_ones(img.value)
        writer = MemmapArrayWriter(path, mmap.offset, img.shape, img_dask.chunksize)
        da.store(img_dask, writer)
        return

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

        warnings.warn(f"Image axes changed", UserWarning, stacklevel=2)

    img = img.sort_axes()
    if img.dtype == "bool":
        img = img.astype(np.uint8)  # tif does not support bool
    imsave_kwargs = _get_ijmeta_from_img(img, update_lut=True)
    imwrite(path, img, **imsave_kwargs)
    return None

def _load_json(s: str) -> dict[str, Any]:
    return json.loads(re.sub("'", '"', s))

def _parse_tifffile(tif: TiffFile) -> ImageMetadata:
    from tifffile import xml2dict

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
    scale: dict[str, float] = {}
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
    if ome_meta := tif.ome_metadata:
        try:
            xml = xml2dict(ome_meta)
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
    if lsm_meta := tif.lsm_metadata:
        scale_unit = "μm"
        if voxel_size_x := lsm_meta.get("VoxelSizeX"):
            scale["x"] = voxel_size_x * 1e6
        if voxel_size_y := lsm_meta.get("VoxelSizeY"):
            scale["y"] = voxel_size_y * 1e6
        if voxel_size_z := lsm_meta.get("VoxelSizeZ"):
            scale["z"] = voxel_size_z * 1e6

    return ImageMetadata(
        axes=axes,
        scale=scale,
        unit=scale_unit,
        metadata=ijmeta,
        labels=labels,
    )

def _get_ijmeta_from_img(img: ImpyArray, update_lut=True):
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
    if scale_unit and scale_unit[0] in ("μ", "\xb5", "\xc2"):
        scale_unit = "\\u00B5" + scale_unit[1:]
    metadata["unit"] = scale_unit

    # set axes in tiff metadata
    metadata["axes"] = str(img.axes).upper()
    if img.ndim > 3:
        metadata["hyperstack"] = True

    return dict(imagej=True, resolution=res, metadata=metadata)
