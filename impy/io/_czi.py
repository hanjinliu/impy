from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from impy.io._registry import IO
from impy.io._utils import ImageData, ImageMetadata

if TYPE_CHECKING:
    from czifile import CziFile

@IO.mark_reader(".czi")
def _(path: str, memmap: bool = False) -> ImageData:
    """The CZI format reader"""
    import czifile

    if memmap:
        raise NotImplementedError("CZI reader does not support memmap mode yet.")
    else:
        out = None
    with czifile.CziFile(path) as czi:
        data = np.squeeze(czi.asarray(out=out))
        meta = _parse_czi_meta(czi)

    return ImageData.from_metadata(data, meta)

def _parse_czi_meta(czi: CziFile) -> ImageMetadata:
    """Parse metadata from a CZI file."""
    axes_all: str = czi.axes
    shape = czi.shape
    axes = "".join(a.lower() for s, a in zip(shape, axes_all) if s > 1)
    try:
        img_meta = czi.metadata(raw=False)["ImageDocument"]["Metadata"]
        scale = {}
        for item in img_meta["Scaling"]["Items"]["Distance"]:
            a = item["Id"].lower()
            if a in axes:
                scale[a] = item.get("Value", 1e-6) * 1e6  # as micron
        try:
            labels = [item["ImageChannelName"] for item in img_meta["Experiment"]["ExperimentBlocks"]["AcquisitionBlock"]["MultiTrackSetup"]["TrackSetup"]["Detectors"]["Detector"]]
        except Exception:
            labels = None
    except KeyError:
        scale = None
        labels = None

    return ImageMetadata(
        axes=axes,
        scale=scale,
        unit="μm",
        metadata=img_meta,
        labels=labels,
    )
