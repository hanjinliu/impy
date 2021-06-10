from __future__ import annotations
from ..utilcls import ImportOnRequest
import numpy as np
from ..labeledarray import LabeledArray
from ..phasearray import PhaseArray
from .mouse import *
napari = ImportOnRequest("napari")

def copy_layer(layer):
    states = layer.as_layer_data_tuple()
    copy = layer.__class__(states[0], **states[1])
    return copy

def iter_layer(viewer, layer_type:str):
    """
    Iterate over layers and yield only certain type of layers.

    Parameters
    ----------
    layer_type : str, {"shape", "image", "point"}
        Type of layer.

    Yields
    -------
    napari.layers
        Layers specified by layer_type
    """        
    if isinstance(layer_type, str):
        layer_type = [layer_type]
    layer_type = tuple(getattr(napari.layers, t) for t in layer_type)
    
    for layer in viewer.layers:
        if isinstance(layer, layer_type):
            yield layer

def iter_selected_layer(viewer, layer_type:str|list[str]):
    if isinstance(layer_type, str):
        layer_type = [layer_type]
    layer_type = tuple(getattr(napari.layers, t) for t in layer_type)
    
    for layer in viewer.layers.selection:
        if isinstance(layer, layer_type):
            yield layer

def front_image(viewer):
    """
    From list of image layers return the most front visible image.
    """        
    front = None
    for img in iter_layer(viewer, "Image"):
        if img.visible:
            front = img # This is ImgArray
    if front is None:
        raise ValueError("There is no visible image layer.")
    return front

def to_labels(layer, labels_shape, zoom_factor=1):
    return layer._data_view.to_labels(labels_shape=labels_shape, zoom_factor=zoom_factor)
    

def make_world_scale(obj):
    scale = []
    for a in obj._axes:
        if a in "zyx":
            scale.append(obj.scale[a])
        elif a == "c":
            pass
        else:
            scale.append(1)
    return scale

def upon_add_layer(event):
    try:
        new_layer = event.sources[0][-1]
    except IndexError:
        return None
    new_layer.translate = new_layer.translate.astype(np.float64)
    if isinstance(new_layer, napari.layers.Shapes):
        @new_layer.bind_key("Alt-A")
        def select_all_shapes(layer):
            layer.selected_data = set(np.arange(layer.nshapes))
    if isinstance(new_layer, napari.layers.Points):
        @new_layer.bind_key("Alt-A")
        def select_all_points(layer):
            layer.selected_data = set(np.arange(layer.data.shape[0]))
            
    new_layer.metadata["init_translate"] = new_layer.translate.copy()
    new_layer.metadata["init_scale"] = new_layer.scale.copy()
        
    return None

def add_labeledarray(viewer, img:LabeledArray, **kwargs):
    chn_ax = img.axisof("c") if "c" in img.axes else None
        
    if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
        kwargs["colormap"] = "hsv"
        kwargs["contrast_limits"] = img.border
    
    scale = make_world_scale(img)
    
    if len(img.history) > 0:
        suffix = "-" + img.history[-1]
    else:
        suffix = ""
    
    name = "No-Name" if img.name is None else img.name
    if chn_ax is not None:
        name = [f"[C{i}]{name}{suffix}" for i in range(img.sizeof("c"))]
    else:
        name = [name + suffix]
    
    layer = viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                             name=name if len(name)>1 else name[0],
                             **kwargs)
    
    viewer.scale_bar.unit = img.scale_unit
    new_axes = [a for a in img.axes if a != "c"]
    # add axis labels to slide bars and image orientation.
    if len(new_axes) >= len(viewer.dims.axis_labels):
        viewer.dims.axis_labels = new_axes
    return layer
