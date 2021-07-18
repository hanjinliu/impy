from __future__ import annotations
from tifffile import TiffFile
import json
import re
import os
import numpy as np
from dask import array as da
from skimage import io
from .._const import MAX_GB

def load_json(s:str):
    return json.loads(re.sub("'", '"', s))

def check_size(path:str):
    size = os.path.getsize(path)/1e9
    if size > MAX_GB:
        raise MemoryError(f"Too large {size:.2f} GB")

def open_tif(path:str, return_img:bool=False, memmap:bool=False):
    with TiffFile(path) as tif:
        ijmeta = tif.imagej_metadata
        series0 = tif.series[0]
    
        pagetag = series0.pages[0].tags
        
        hist = []
        if ijmeta is None:
            ijmeta = {}
        
        ijmeta.pop("ROI", None)
        
        if "Info" in ijmeta.keys():
            try:
                infodict = load_json(ijmeta["Info"])
            except:
                infodict = {}
            if "impyhist" in infodict.keys():
                hist = infodict["impyhist"].split("->")
        
        try:
            axes = series0.axes.lower()
        except:
            axes = None
        
        tags = {v.name: v.value for v in pagetag.values()}
        out = {"axes": axes, "ijmeta": ijmeta, "history": hist, "tags": tags}
        if return_img:
            if memmap:
                out["image"] = tif.asarray(out="memmap")
            else:
                check_size(path)
                out["image"] = tif.asarray()

    return out

def open_mrc(path:str, return_img:bool=False, memmap:bool=False):
    import mrcfile
    if memmap:
        open_func = mrcfile.mmap
    else:
        open_func = mrcfile.open
        check_size(path)
        
    with open_func(path) as mrc:
        ijmeta = {"unit": "nm"}
        ndim = len(mrc.voxel_size.item())
        if ndim == 3:
            axes = "zyx"
            ijmeta["spacing"] = mrc.voxel_size.z/10
        elif ndim == 2:
            axes = "yx"
        else:
            raise RuntimeError(f"ndim = {ndim} not supported")
            
        tags = {}
        tags["XResolution"] = [1, mrc.voxel_size.x/10]
        tags["YResolution"] = [1, mrc.voxel_size.y/10]
        
        out = {"axes": axes, "ijmeta": ijmeta, "history": [], "tags": tags}
        if return_img:
            out["image"] = mrc.data
    
    return out

def open_as_dask(path:str, chunks):
    meta, img = open_img(path, memmap=True)
    axes = meta["axes"]
    if chunks == "default":
        set_chunks = lambda i: img.shape[i] if axes[i] in "yx" else "auto"
        chunks = tuple(map(set_chunks, range(img.ndim)))
    img = da.from_array(img, chunks=chunks)
    return meta, img


def open_img(path, memmap:bool=False):
    _, fext = os.path.splitext(os.path.basename(path))
    if fext in (".tif", ".tiff"):
        meta = open_tif(path, True, memmap=memmap)
        img = meta.pop("image")
    elif fext in (".mrc", ".rec"):
        meta = open_mrc(path, True, memmap=memmap)
        img = meta.pop("image")
    else:
        img = io.imread(path)
        if fext in (".png", ".jpg") and img.ndim == 3 and img.shape[-1] <= 4:
            meta = {"axes":"yxc", "ijmeta":{}, "history":[]}
        else:
            meta = {"axes":None, "ijmeta":{}, "history":[]}
    
    return meta, img

def get_scale_from_meta(meta:dict):
    spacing = meta["ijmeta"].get("spacing", 1.0)
    scale = dict()
    try:
        tags = meta["tags"]
        xres = tags["XResolution"]
        yres = tags["YResolution"]
        dx = xres[1]/xres[0]
        dy = yres[1]/yres[0]
    except KeyError:
        dx = dy = spacing
    
    scale["x"] = dx
    scale["y"] = dy
    # read z scale if needed
    if "z" in meta["axes"]:
        scale["z"] = spacing
        
    return scale

def get_imsave_meta_from_img(img, update_lut=True):
    metadata = img.metadata.copy()
    if update_lut:
        lut_min, lut_max = np.percentile(img, [1, 99])
        metadata.update({"min": lut_min, 
                         "max": lut_max})
    # set lateral scale
    try:
        res = (1/img.scale["x"], 1/img.scale["y"])
    except Exception:
        res = None
    # set z-scale
    if "z" in img.axes:
        metadata["spacing"] = img.scale["z"]
    # add history to Info
    try:
        info = load_json(metadata["Info"])
    except:
        info = {}
    info["impyhist"] = "->".join([img.name] + img.history)
    metadata["Info"] = str(info)
    # set axes in tiff metadata
    metadata["axes"] = str(img.axes).upper()
    if img.ndim > 3:
        metadata["hyperstack"] = True
    
    return dict(imagej=True, resolution=res, metadata=metadata)