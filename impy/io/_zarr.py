from __future__ import annotations

import os
import numpy as np
from impy.io._registry import IO
from impy.io._utils import get_channel_labels, ImpyArray, ImageData

from impy.axes import ImageAxesError


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
    f.attrs["labels"] = get_channel_labels(img.axes)
    if lazy:
        img.value.to_zarr(url=os.path.join(path, "data"))
    else:
        z = img.value
        f.array("data", z)
    return None
