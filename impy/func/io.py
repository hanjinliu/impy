from __future__ import annotations
from tifffile import TiffFile
import json
import re
import os
from skimage import io

def load_json(s:str):
    return json.loads(re.sub("'", '"', s))

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
                out["image"] = tif.asarray()

    return out

def open_mrc(path:str, return_img:bool=False, memmap:bool=False):
    import mrcfile
    if memmap:
        open_func = mrcfile.mmap
    else:
        open_func = mrcfile.open
        
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