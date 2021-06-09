from __future__ import annotations
from ..utilcls import ImportOnRequest
import numpy as np
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
    if isinstance(new_layer, napari.layers.Image):
        new_layer.translate = new_layer.translate.astype(np.float64)
        new_layer.mouse_drag_callbacks.append(drag_translation)
        new_layer.mouse_wheel_callbacks.append(wheel_resize)
        new_layer.metadata["init_translate"] = new_layer.translate.copy()
        new_layer.metadata["init_scale"] = new_layer.scale.copy()
        
    return None
