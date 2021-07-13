from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..arrays import *
from ..frame import *
import napari

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
        _text_bound_init(new_layer)
        mean_scale = np.mean(new_layer.scale[-2:])
        new_layer.current_edge_width = new_layer.scale[-1] * 2.0
        new_layer.current_face_color = [1, 1, 1, 0]
        new_layer.current_edge_color = "#dd23cb"
        new_layer._rotation_handle_length = 20/mean_scale
        
    if isinstance(new_layer, napari.layers.Points):
        _text_bound_init(new_layer)
    
    if isinstance(new_layer, napari.layers.Labels):
        _text_bound_init(new_layer)
            
    new_layer.metadata["init_translate"] = new_layer.translate.copy()
    new_layer.metadata["init_scale"] = new_layer.scale.copy()
        
    return None

def _text_bound_init(new_layer):
    @new_layer.bind_key("Alt-A")
    def select_all(layer):
        layer.selected_data = set(np.arange(len(layer.data)))
        
    # txt_manager = napari.layers.utils.text.TextManager
    # init_prop = {"text": np.zeros(0, dtype="<U12")}
    # new_layer._text = txt_manager("{text}", 0, properties=init_prop,
    #                               color="yellow", translation=[-8, 0], size=8)
    # new_layer._properties, new_layer._property_choices = new_layer._prepare_properties(
    #                         init_prop, init_prop, save_choices=True)
    # new_layer.current_properties = {k: np.asarray([v[0]]) for k, v in new_layer._property_choices.items()}
    # use refresh text

def add_labeledarray(viewer, img:LabeledArray, **kwargs):
    chn_ax = img.axisof("c") if "c" in img.axes else None
        
    if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
        kwargs["colormap"] = "hsv"
        kwargs["contrast_limits"] = img.border
    elif img.dtype.kind == "c" and  not "colormap" in kwargs.keys():
        kwargs["colormap"] = "plasma"
    
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
    
    if img.dtype.kind == "c":
        img = np.abs(img)
    layer = viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                             name=name if len(name)>1 else name[0],
                             **kwargs)
    
    viewer.scale_bar.unit = img.scale_unit
    new_axes = [a for a in img.axes if a != "c"]
    # add axis labels to slide bars and image orientation.
    if len(new_axes) >= len(viewer.dims.axis_labels):
        viewer.dims.axis_labels = new_axes
    return layer

def get_viewer_scale(viewer):
    return {a: r[2] for a, r in zip(viewer.dims.axis_labels, viewer.dims.range)}

def layer_to_impy_object(viewer, layer):
    """
    Convert layer to real data.

    Parameters
    ----------
    layer : napari.layers.Layer
        Input layer.

    Returns
    -------
    ImgArray, Label, MarkerFrame or TrackFrame, or Shape features.
    """ 
    data = layer.data
    axes = "".join(viewer.dims.axis_labels)
    scale = get_viewer_scale(viewer)
    if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
        # manually drawn ones are np.ndarray, need conversion
        ndim = data.ndim
        axes = axes[-ndim:]
        if type(data) is np.ndarray:
            if isinstance(layer, napari.layers.Image):
                data = ImgArray(data, name=layer.name, axes=axes, dtype=layer.data.dtype)
            else:
                data = Label(data, name=layer.name, axes=axes)
            data.set_scale({k: v for k, v in scale.items() if k in axes})
        return data
    elif isinstance(layer, napari.layers.Shapes):
        return data
    elif isinstance(layer, napari.layers.Points):
        ndim = data.shape[1]
        axes = axes[-ndim:]
        df = MarkerFrame(data, columns=layer.metadata.get("axes", axes))
        df.set_scale(layer.metadata.get("scale", 
                                        {k: v for k, v in scale.items() if k in axes}))
        return df.as_standard_type()
    elif isinstance(layer, napari.layers.Tracks):
        ndim = data.shape[1]
        axes = axes[-ndim:]
        df = TrackFrame(data, columns=layer.metadata.get("axes", axes))
        df.set_scale(layer.metadata.get("scale", 
                                        {k: v for k, v in scale.items() if k in axes}))
        return df.as_standard_type()
    else:
        raise NotImplementedError(type(layer))


class ColorCycle:
    def __init__(self, cmap="rainbow") -> None:
        self.cmap = plt.get_cmap(cmap, 16)
        self.color_id = 0
    
    def __call__(self):
        """return next colormap"""
        self.color_id += 1
        return list(self.cmap(self.color_id * (self.cmap.N//2+1) % self.cmap.N))