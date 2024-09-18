from __future__ import annotations

from impy.io._registry import IO
from impy.io._utils import ImageData

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
