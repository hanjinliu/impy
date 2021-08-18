import numpy as np
from napari.layers.utils._link_layers import link_layers, unlink_layers
import napari

from .utils import *
from .widgets import DuplicateDialog, ImageProjectionDialog, LabelProjectionDialog

from ..arrays import LabeledArray
from ..core import array as ip_array

# Shift, Control, Alt, Meta, Up, Down, Left, Right, PageUp, PageDown, Insert, 
# Delete, Home, End, Escape, Backspace, F1, F2, F3, F4, F5, F6, F7, F8, F9, F10,
# F11, F12, Space, Enter, Tab

KEYS = {"focus_next": "]",
        "focus_previous": "[",
        "hide_others": "Control-Shift-A",
        "link_selected_layers": "Control-G",
        "unlink_selected_layers": "Control-Shift-G",
        "layers_to_labels": "Alt-L",
        "crop": "Control-Shift-X",
        "reslice": "/",
        "to_front": "Control-Shift-F",
        "reset_view": "Control-Shift-R",
        "proj": "Control-P",
        "duplicate_layer": "Control-Shift-D",
        }

__all__ = list(KEYS.keys())

def bind_key(func):
    return napari.Viewer.bind_key(KEYS[func.__name__])(func)
    
@bind_key
def focus_next(viewer:"napari.Viewer"):
    _change_focus(viewer, 1)
    return None

@bind_key
def focus_previous(viewer:"napari.Viewer"):
    _change_focus(viewer, -1)
    return None

def _change_focus(viewer:"napari.Viewer", ind:int):
    # assert one Shapes or Points layer is selected
    selected_layer = list(viewer.layers.selection)
    if len(selected_layer) != 1:
        return None
    selected_layer = selected_layer[0]
    if not isinstance(selected_layer, (napari.layers.Shapes, napari.layers.Points)):
        return None

    # check if one shape/point is selected
    selected_data = list(selected_layer.selected_data)
    if len(selected_data) != 1:
        return None
    selected_data = selected_data[0]
    
    # determine next/previous index/data to select
    ndata = len(selected_layer.data)
    next_to_select = (selected_data + ind) % ndata
    next_data = np.atleast_2d(selected_layer.data[next_to_select])
    
    # update camera    
    scale = selected_layer.scale
    center = np.mean(next_data, axis=0) * scale
    current_zoom = viewer.camera.zoom
    current_center = viewer.camera.center
    next_center = current_center[:-2] + center[-2:]
    viewer.dims.current_step = list(next_data[0, :-2].astype(np.int64)) + [0, 0]
    
    # TODO: Currently the value of "zoom" is inconsistent with unknown reason.
    # I don't know if this problem will be fixed in 0.4.11.
    viewer.camera.center = next_center
    viewer.camera.zoom = current_zoom
    
    selected_layer.selected_data = {next_to_select}
    selected_layer._set_highlight()
    return None
    
    
@bind_key
def hide_others(viewer:"napari.Viewer"):
    """
    Make selected layers visible and others invisible. 
    """
    selected = viewer.layers.selection
    visibility_old = [layer.visible for layer in viewer.layers]
    visibility_new = [layer in selected for layer in viewer.layers]
    if visibility_old != visibility_new:
        for layer, vis in zip(viewer.layers, visibility_new):
            layer.visible = vis
    else:
        for layer in viewer.layers:
            layer.visible = True

@bind_key
def link_selected_layers(viewer:"napari.Viewer"):
    """
    Link selected layers.
    """
    link_layers(viewer.layers.selection)
    
@bind_key
def unlink_selected_layers(viewer:"napari.Viewer"):
    """
    Unlink selected layers.
    """
    unlink_layers(viewer.layers.selection)

@bind_key
def to_front(viewer:"napari.Viewer"):
    """
    Let selected layers move to front.
    """
    not_selected_index = [i for i, l in enumerate(viewer.layers) 
                          if l not in viewer.layers.selection]
    viewer.layers.move_multiple(not_selected_index, 0)
    
@bind_key
def reset_view(viewer:"napari.Viewer"):
    """
    Reset translate/scale parameters to the initial value.
    """    
    for layer in viewer.layers.selection:
        # layer.translte[:] = 0. did not work
        layer.translate -= (layer.translate - layer.metadata["init_translate"])
        layer.scale = layer.metadata["init_scale"]

@bind_key
def layers_to_labels(viewer:"napari.Viewer"):
    """
    Convert manually drawn shapes to labels and store it.
    """        
    # determine destinations.
    destinations = [l.data for l in iter_selected_layer(viewer, "Image")]
    if len(destinations) == 0:
        destinations = [front_image(viewer).data]
    
    for dst in destinations:
        # check zoom_factors
        d = viewer.dims
        scale = {a: r[2] for a, r in zip(d.axis_labels, d.range)}
        # TODO: image translation
        # zoom_factors = layer.scale[-2:]/shape_layer_scale[-2:]
        zoom_factors = [scale[a]/dst.scale[a] for a in "yx"]
        if np.unique(zoom_factors).size == 1:
            zoom_factor = zoom_factors[0]
        else:
            raise ValueError("Scale mismatch in images and napari world.")
        
        # make labels from selected layers
        shapes = [to_labels(layer, dst.shape, zoom_factor=zoom_factor) 
                  for layer in iter_selected_layer(viewer, "Shapes")]
        labels = [layer.data for layer in iter_selected_layer(viewer, "Labels")]
        
        if len(shapes) > 0 and len(labels) > 0:
            viewer.status = "Both Shapes and Labels were selected"
            return None
        elif len(shapes) > 0:
            labelout = np.max(shapes, axis=0)
            viewer.add_labels(labelout, opacity=0.3, scale=list(scale.values()), name="Labeled by Shapes")
        elif len(labels) > 0:
            labelout = np.max(labels, axis=0)
        else:
            return None
        
        # append labels to each destination
        if hasattr(dst, "labels"):
            print(f"Label already exist in {dst}. Overlapped.")
            del dst.labels
        dst.append_label(labelout)
        
    return None

@bind_key
def crop(viewer:"napari.Viewer"):
    """
    Crop images with (rotated) rectangle shapes.
    """        
    if viewer.dims.ndisplay == 3:
        viewer.status = "Cannot crop in 3D mode."
    imglist = list(iter_selected_layer(viewer, "Image"))
    if len(imglist) == 0:
        imglist = [front_image(viewer)]
    
    ndim = np.unique([shape_layer.ndim for shape_layer 
                      in iter_selected_layer(viewer, "Shapes")])
    if len(ndim) > 1:
        viewer.status = "Cannot crop using Shapes layers with different number of dimensions."
    else:
        ndim = ndim[0]
    
    if ndim == viewer.dims.ndim:
        active_plane = list(viewer.dims.order[-2:])
    else:
        active_plane = [-2, -1]
    dims2d = "".join(viewer.dims.axis_labels[i] for i in active_plane)
    rects = []
    for shape_layer in iter_selected_layer(viewer, "Shapes"):
        for i in range(shape_layer.nshapes):
            shape = shape_layer.data[i]
            type_ = shape_layer.shape_type[i]
            if type_ == "rectangle":
                rects.append((shape[:, active_plane],            # shape = float pixel
                              shape_layer.scale[active_plane],
                              _get_property(shape_layer, i))
                             )

    for rect, shape_layer_scale, prop in rects:
        if np.any(np.abs(rect[0] - rect[1])<1e-5):
            crop_func = _crop_rectangle
        else:
            crop_func = _crop_rotated_rectangle
        
        for layer in imglist:
            factor = layer.scale[active_plane]/shape_layer_scale
            _name = prop + layer.name
            layer = viewer.add_layer(copy_layer(layer))
            dr = layer.translate[active_plane] / layer.scale[active_plane]
            newdata, relative_translate = \
                crop_func(layer.data, rect/factor - dr, dims2d)
            if newdata.size <= 0:
                continue
            
            if not isinstance(newdata, (LabeledArray, LazyImgArray)):
                scale = get_viewer_scale(viewer)
                axes = "".join(viewer.dims.axis_labels)
                newdata = ip_array(newdata, axes=axes)
                newdata.set_scale(**scale)
                
            newdata.dirpath = layer.data.dirpath
            newdata.metadata = layer.data.metadata
            newdata.name = layer.data.name
                
            layer.data = newdata
            translate = layer.translate
            translate[active_plane] += relative_translate * layer.scale[active_plane]
            layer.translate = translate
            layer.name = _name
            layer.metadata.update({"init_translate": layer.translate, 
                                   "init_scale": layer.scale})
            
    # remove original images
    [viewer.layers.remove(img) for img in imglist]
    return None

@bind_key
def reslice(viewer:"napari.Viewer"):
    """
    2D Reslice with currently selected lines/paths and images.
    """
    if viewer.dims.ndisplay == 3:
        viewer.status = "Cannot reslice in 3D mode."
    imglist = list(iter_selected_layer(viewer, "Image"))
    
    ndim = np.unique([shape_layer.ndim for shape_layer 
                      in iter_selected_layer(viewer, "Shapes")])
    if len(ndim) > 1:
        viewer.status = "Cannot crop using Shapes layers with different number of dimensions."
    else:
        ndim = ndim[0]
        
    if ndim == viewer.dims.ndim == 3:
        active_plane = [-3, -2, -1]
    else:
        active_plane = [-2, -1]
        
    if len(imglist) == 0:
        imglist = [front_image(viewer)]
    
    paths = []
    for shape_layer in iter_selected_layer(viewer, "Shapes"):
        for shape, type_ in zip(shape_layer.data, shape_layer.shape_type):
            if type_ in ("line", "path"):
                paths.append((shape, shape_layer.scale)) # shape = float pixel
    out = []
    for path, shape_layer_scale in paths:        
        for layer in imglist:
            factor = layer.scale[active_plane]/shape_layer_scale[active_plane]
            dr = layer.translate[active_plane] / layer.scale[active_plane]
            out_ = layer.data.reslice(path[:,active_plane]/factor - dr)
            out.append(out_)
    
    viewer.window.results = out
    
    return None

@bind_key
def proj(viewer:"napari.Viewer"):
    """
    Projection
    """
    layer = get_a_selected_layer(viewer)
    
    data = layer.data
    kwargs = {}
    kwargs.update({"scale": layer.scale[-2:], 
                    "translate": layer.translate[-2:],
                    "blending": layer.blending,
                    "opacity": layer.opacity,
                    "ndim": 2,
                    "name": f"[Proj]{layer.name}"})
    if isinstance(layer, napari.layers.Image):
        if layer.data.ndim < 3:
            return None
        dlg = ImageProjectionDialog(viewer, layer)
        dlg.exec_()
        
    elif isinstance(layer, napari.layers.Labels):
        if layer.data.ndim < 3:
            return None
        dlg = LabelProjectionDialog(viewer, layer)
        dlg.exec_()
            
    elif isinstance(layer, napari.layers.Shapes):
        data = [d[:,-2:] for d in data]
        
        if layer.nshapes > 0:
            for k in ["face_color", "edge_color", "edge_width"]:
                kwargs[k] = getattr(layer, k)
        else:
            data = None
        kwargs["ndim"] = 2
        kwargs["shape_type"] = layer.shape_type
        viewer.add_shapes(data, **kwargs)
        
    elif isinstance(layer, napari.layers.Points):
        data = data[:, -2:]
        for k in ["face_color", "edge_color", "size", "symbol"]:
            kwargs[k] = getattr(layer, k, None)
        kwargs["size"] = layer.size[:,-2:]
        if len(data) == 0:
            data = None
        kwargs["ndim"] = 2
        viewer.add_points(data, **kwargs)
        
    elif isinstance(layer, napari.layers.Tracks):
        data = data[:, [0, -2, -1]] # p, y, x axes
        viewer.add_tracks(data, **kwargs)
        
    else:
        raise NotImplementedError(type(layer))
    
@bind_key
def duplicate_layer(viewer:"napari.Viewer"):
    """
    Duplicate the selected layer.
    """
    layer = get_a_selected_layer(viewer)
    
    if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
        dlg = DuplicateDialog(viewer, layer)
        dlg.exec_()
    else:
        new_layer = copy_layer(layer)
        viewer.add_layer(new_layer)
    return None

def _crop_rotated_rectangle(img, crds, dims):
    translate = np.min(crds, axis=0)
    
    # check is sorted
    ids = [img.axisof(a) for a in dims]
    if sorted(ids) == ids:
        cropped_img = img.rotated_crop(crds[1], crds[0], crds[2], dims=dims)
    else:
        crds = np.fliplr(crds)
        cropped_img = img.rotated_crop(crds[3], crds[0], crds[2], dims=dims)
    
    return cropped_img, translate

def _crop_rectangle(img, crds, dims):
    start = crds[0]
    end = crds[2]
    sl = []
    translate = np.empty(2)
    for i in [0, 1]:
        sl0 = sorted([start[i], end[i]])
        x0 = max(int(sl0[0]), 0)
        x1 = min(int(sl0[1]), img.sizeof(dims[i]))
        sl.append(f"{dims[i]}={x0}:{x1}")
        translate[i] = x0
    
    area_to_crop = ";".join(sl)
    
    cropped_img = img[area_to_crop]
    return cropped_img, translate

def _get_property(layer, i):
    try:
        prop = layer.properties["text"][i]
    except (KeyError, IndexError):
        prop = ""
    if prop != "":
        prop = prop + " of "
    return prop