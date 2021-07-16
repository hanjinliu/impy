from ..arrays import LabeledArray
from ..core import array as ip_array
from .utils import *
import numpy as np
from napari.layers.utils._link_layers import link_layers, unlink_layers
import napari

KEYS = {"hide_others": "Control-Shift-A",
        "link_selected_layers": "Control-G",
        "unlink_selected_layers": "Control-Shift-G",
        "layers_to_labels": "Alt-L",
        "crop": "Control-Shift-X",
        "reslice": "/",
        "to_front": "Control-Shift-F",
        "reset_view": "Control-Shift-R",
        "add_new_shape": "S",
        "proj": "Control-P",
        "duplicate_layer": "Control-Shift-D",
        }

__all__ = list(KEYS.keys())

def bind_key(func):
    return napari.Viewer.bind_key(KEYS[func.__name__])(func)

@bind_key
def hide_others(viewer:napari.Viewer):
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
def link_selected_layers(viewer:napari.Viewer):
    """
    Link selected layers.
    """
    link_layers(viewer.layers.selection)
    
@bind_key
def unlink_selected_layers(viewer:napari.Viewer):
    """
    Unlink selected layers.
    """
    unlink_layers(viewer.layers.selection)

@bind_key
def to_front(viewer:napari.Viewer):
    """
    Let selected layers move to front.
    """
    not_selected_index = [i for i, l in enumerate(viewer.layers) 
                          if l not in viewer.layers.selection]
    viewer.layers.move_multiple(not_selected_index, 0)
    
@bind_key
def reset_view(viewer:napari.Viewer):
    """
    Reset translate/scale parameters to the initial value.
    """    
    for layer in viewer.layers.selection:
        # layer.translte[:] = 0. did not work
        layer.translate -= (layer.translate - layer.metadata["init_translate"])
        layer.scale = layer.metadata["init_scale"]

@bind_key
def layers_to_labels(viewer:napari.Viewer):
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
def crop(viewer:napari.Viewer):
    """
    Crop images with (rotated) rectangle shapes.
    """        
    # BUG: wrong result in XZ crop
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
        for shape, type_ in zip(shape_layer.data, shape_layer.shape_type):
            if type_ == "rectangle":
                rects.append((shape[:, active_plane], 
                              shape_layer.scale[active_plane])) # shape = float pixel
                
    for rect, shape_layer_scale in rects:
        if np.any(np.abs(rect[0] - rect[1])<1e-5):
            crop_func = _crop_rectangle
        else:
            crop_func = _crop_rotated_rectangle
        
        for layer in imglist:
            factor = layer.scale[active_plane]/shape_layer_scale
            _dirpath = layer.data.dirpath
            _metadata = layer.data.metadata
            _name = layer.data.name
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
                
            newdata.dirpath =_dirpath
            newdata.metadata = _metadata
            newdata.name = _name
            # Try to compute for now too avoid response being too slow.
            if isinstance(newdata, LazyImgArray):
                try:
                    newdata = newdata.release()
                except MemoryError:
                    pass
                
            layer.data = newdata
            translate = layer.translate
            translate[active_plane] += relative_translate * layer.scale[active_plane]
            layer.translate = translate
            layer.metadata.update({"init_translate": layer.translate, 
                                   "init_scale": layer.scale})
            
    # remove original images
    [viewer.layers.remove(img) for img in imglist]
    return None

@bind_key
def reslice(viewer:napari.Viewer):
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
def proj(viewer:napari.Viewer):
    """
    Projection
    """
    layers = list(viewer.layers.selection)
    for layer in layers:
        data = layer.data
        kwargs = {}
        kwargs.update({"scale": layer.scale[-2:], 
                       "translate": layer.translate[-2:],
                       "blending": layer.blending,
                       "opacity": layer.opacity,
                       "ndim": 2,
                       "name": layer.name+"-proj"})
        if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
            raise TypeError("Projection not supported.")
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
            data = data[:, [0,-2,-1]]
            viewer.add_tracks(data, **kwargs)
        else:
            raise NotImplementedError(type(layer))
        
@bind_key
def duplicate_layer(viewer:napari.Viewer):
    """
    Duplicate selected layer(s).
    """
    [viewer.add_layer(copy_layer(layer)) for layer in list(viewer.layers.selection)]

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
        x0 = int(sl0[0])+1
        x1 = int(sl0[1])+1
        sl.append(f"{dims[i]}={x0}:{x1}")
        translate[i] = x0
    
    area_to_crop = ";".join(sl)
    
    cropped_img = img[area_to_crop]
    return cropped_img, translate