from __future__ import annotations

from typing import TYPE_CHECKING
from impy.io._registry import IO
from impy.io._utils import ImageData, ImageMetadata

if TYPE_CHECKING:
    from nd2 import ND2File

@IO.mark_reader(".nd2")
def _(path: str, memmap: bool = False):
    import nd2

    with nd2.ND2File(path) as f:
        if not memmap:
            image = f.asarray()
        else:
            image = f.to_dask()
        meta = _parse_nd2_metadata(f)

    return ImageData.from_metadata(image, meta)

@IO.mark_header_reader(".nd2")
def _(path: str):
    import nd2

    with nd2.ND2File(path) as f:
        meta = _parse_nd2_metadata(f)
    return meta

def _parse_nd2_metadata(f: ND2File) -> ImageMetadata:
    from dataclasses import is_dataclass, asdict

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

    return ImageMetadata(
        axes=axes,
        scale=scale,
        unit="μm",
        metadata=metadata,
        labels=labels,
    )
