from __future__ import annotations
import numpy as np
import napari

from ..arrays import *
from ..frame import *
from .._const import Const

def copy_layer(layer):
    args, kwargs, *_ = layer.as_layer_data_tuple()
    # linear interpolation is valid only in 3D mode.
    if kwargs["interpolation"] == "linear":
        kwargs = kwargs.copy()
        kwargs["interpolation"] = "nearest"
    copy = layer.__class__(args, **kwargs)
    return copy

def iter_layer(viewer:"napari.Viewer", layer_type:str):
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

def iter_selected_layer(viewer:"napari.Viewer", layer_type:str|list[str]):
    if isinstance(layer_type, str):
        layer_type = [layer_type]
    layer_type = tuple(getattr(napari.layers, t) for t in layer_type)
    
    for layer in viewer.layers.selection:
        if isinstance(layer, layer_type):
            yield layer

def front_image(viewer:"napari.Viewer"):
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

def to_labels(layer:napari.layers.Shapes, labels_shape, zoom_factor=1):
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
        # change default
        new_layer.current_edge_width = 2.0 # unit is pixel here
        new_layer.current_face_color = [1, 1, 1, 0]
        new_layer.current_edge_color = "#dd23cb"
        new_layer._rotation_handle_length = 20/np.mean(new_layer.scale[-2:])
        @new_layer.bind_key("Left")
        def left(layer):
            _translate_shape(layer, -1, -1)
            
        @new_layer.bind_key("Right")
        def right(layer):
            _translate_shape(layer, -1, 1)
            
        @new_layer.bind_key("Up")
        def up(layer):
            _translate_shape(layer, -2, -1)
            
        @new_layer.bind_key("Down")
        def down(layer):
            _translate_shape(layer, -2, 1)
            
    elif isinstance(new_layer, napari.layers.Points):
        _text_bound_init(new_layer)
        new_layer.current_face_color = [1, 1, 1, 0]
        new_layer.current_edge_color = "#68cbc3ff"
        new_layer.edge_width = 3
        new_layer.current_size = 4
                
    new_layer.metadata["init_translate"] = new_layer.translate.copy()
    new_layer.metadata["init_scale"] = new_layer.scale.copy()
        
    return None


def image_tuple(input:"napari.layers.Image", out:ImgArray, translate="inherit", **kwargs):
    data = input.data
    scale = make_world_scale(data)
    if out.dtype.kind == "c":
        out = np.abs(out)
    contrast_limits = [float(x) for x in out.range]
    if data.ndim == out.ndim:
        if isinstance(translate, str) and translate == "inherit":
            translate = input.translate
    elif data.ndim > out.ndim:
        if isinstance(translate, str) and translate == "inherit":
            translate = [input.translate[i] for i in range(data.ndim) if data.axes[i] in out.axes]
        scale = [scale[i] for i in range(data.ndim) if data.axes[i] in out.axes]
    else:
        if isinstance(translate, str) and translate == "inherit":
            translate = [0.0] + list(input.translate)
        scale = [1.0] + list(scale)
    kw = dict(scale=scale, colormap=input.colormap, translate=translate,
              blending=input.blending, contrast_limits=contrast_limits)
    kw.update(kwargs)
    return (out, kw, "image")


def label_tuple(input:"napari.layers.Labels", out:Label, translate="inherit", **kwargs):
    data = input.data
    scale = make_world_scale(data)
    if isinstance(translate, str) and translate == "inherit":
            translate = input.translate
    kw = dict(opacity=0.3, scale=scale, translate=translate)
    kw.update(kwargs)
    return (out, kw, "labels")

def _translate_shape(layer, ind, direction):
    data = layer.data
    selected = layer.selected_data
    for i in selected:
        data[i][:, ind] += direction
    layer.data = data
    layer.selected_data = selected
    layer._set_highlight()
    return None

def _text_bound_init(new_layer):
    @new_layer.bind_key("Alt-A")
    def select_all(layer):
        layer.selected_data = set(np.arange(len(layer.data)))
        layer._set_highlight()
    
    @new_layer.bind_key("Control-Shift-<")
    def size_down(layer):
        if new_layer.text.size > 4:
            new_layer.text.size -= 1.0
        else:
            new_layer.text.size *= 0.8
    
    @new_layer.bind_key("Control-Shift->")
    def size_up(layer):
        if new_layer.text.size < 4:
            new_layer.text.size += 1.0
        else:
            new_layer.text.size /= 0.8
            
    n_obj = len(new_layer.data)
    
    if n_obj == 0 or new_layer.properties == {}:
        new_layer.current_properties = {"text": np.array([""], dtype="<U32")}
        new_layer.properties = {"text": np.array([""]*n_obj, dtype="<U32")}
        new_layer.text = "{text}"
        new_layer.text.size = 6.0 * Const["FONT_SIZE_FACTOR"]
        new_layer.text.color = "#dd23cb"
        new_layer.text.anchor = "upper_left"
    else:
        pass
    

def add_labeledarray(viewer:"napari.Viewer", img:LabeledArray, **kwargs):
    chn_ax = img.axisof("c") if "c" in img.axes else None
        
    if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
        kwargs["colormap"] = "hsv"
        kwargs["contrast_limits"] = img.border
    elif img.dtype.kind == "c" and  not "colormap" in kwargs.keys():
        kwargs["colormap"] = "plasma"
    
    scale = make_world_scale(img)
    
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = "No-Name" if img.name is None else img.name
        if chn_ax is not None:
            name = [f"[C{i}]{name}" for i in range(img.sizeof("c"))]
        else:
            name = [name]
    
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

def add_labels(viewer:"napari.Viewer", labels:Label, opacity:float=0.3, name:str|list[str]=None, 
               **kwargs):
    scale = make_world_scale(labels)
    # prepare label list
    if "c" in labels.axes:
        lbls = labels.split("c")
    else:
        lbls = [labels]
    
    # prepare name list
    if isinstance(name, list):
        names = [f"[L]{n}" for n in name]
    elif isinstance(name, str):
        names = [f"[L]{name}"] * len(lbls)
    else:
        names = [labels.name]
        
    kw = dict(opacity=opacity, scale=scale, name=name)
    kw.update(kwargs)
    
    out_layers = []
    for lbl, name in zip(lbls, names):
        layer = viewer.add_labels(lbl.value, **kw)
        out_layers.append(layer)
    return out_layers

def add_dask(viewer:"napari.Viewer", img:LazyImgArray, **kwargs):
    chn_ax = img.axisof("c") if "c" in img.axes else None
                
    scale = make_world_scale(img)
    
    if "contrast_limits" not in kwargs.keys():
        # contrast limits should be determined quickly.
        leny, lenx = img.shape[-2:]
        sample = img.img[..., ::leny//min(10, leny), ::lenx//min(10, lenx)]
        kwargs["contrast_limits"] = [float(sample.min().compute()), 
                                        float(sample.max().compute())]

    name = "No-Name" if img.name is None else img.name

    if chn_ax is not None:
        name = [f"[Lazy][C{i}]{name}" for i in range(img.sizeof("c"))]
    else:
        name = ["[Lazy]" + name]

    layer = viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                                    name=name if len(name)>1 else name[0], **kwargs)
    viewer.scale_bar.unit = img.scale_unit
    new_axes = [a for a in img.axes if a != "c"]
    # add axis labels to slide bars and image orientation.
    if len(new_axes) >= len(viewer.dims.axis_labels):
        viewer.dims.axis_labels = new_axes
    return layer

def add_points(viewer:"napari.Viewer", points, **kwargs):
    if isinstance(points, MarkerFrame):
        scale = make_world_scale(points)
        points = points.get_coords()
    else:
        scale=None
    
    if "c" in points._axes:
        pnts = points.split("c")
    else:
        pnts = [points]
        
    for each in pnts:
        metadata = {"axes": str(each._axes), "scale": each.scale}
        kw = dict(size=3.2, face_color=[0,0,0,0], metadata=metadata, edge_color=viewer.window.cmap())
        kw.update(kwargs)
        viewer.add_points(each.values, scale=scale, **kw)
        
    return None

def add_tracks(viewer:"napari.Viewer", track:TrackFrame, **kwargs):
    if "c" in track._axes:
        track_list = track.split("c")
    else:
        track_list = [track]
        
    scale = make_world_scale(track[[a for a in track._axes if a != Const["ID_AXIS"]]])
    for tr in track_list:
        metadata = {"axes": str(tr._axes), "scale": tr.scale}
        viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
    
    return None

def add_paths(viewer:"napari.Viewer", paths:PathFrame, **kwargs):
    if "c" in paths._axes:
        path_list = paths.split("c")
    else:
        path_list = [paths]
        
    scale = make_world_scale(paths[[a for a in paths._axes if a != Const["ID_AXIS"]]])
    kw = {"edge_color":"lime", "edge_width":0.3, "shape_type":"path"}
    kw.update(kwargs)

    for path in path_list:
        metadata = {"axes": str(path._axes), "scale": path.scale}
        paths = [single_path.values for single_path in path.split(Const["ID_AXIS"])]
        viewer.add_shapes(paths, scale=scale, metadata=metadata, **kw)
    
    return None

def add_table(viewer:"napari.Viewer", data=None, columns=None, name=None):
    from .widgets import TableWidget
    table = TableWidget(viewer, data, columns=columns, name=name)
    viewer.window.add_dock_widget(table, area="right", name=table.name)
    return table

def get_viewer_scale(viewer:"napari.Viewer"):
    return {a: r[2] for a, r in zip(viewer.dims.axis_labels, viewer.dims.range)}

def layer_to_impy_object(viewer:"napari.Viewer", layer):
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
        if type(data) is np.ndarray:
            ndim = data.ndim
            axes = axes[-ndim:]
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
        import matplotlib.pyplot as plt
        self.cmap = plt.get_cmap(cmap, 16)
        self.color_id = 0
    
    def __call__(self):
        """return next colormap"""
        self.color_id += 1
        return list(self.cmap(self.color_id * (self.cmap.N//2+1) % self.cmap.N))

