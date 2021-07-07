from ..arrays import LabeledArray
from ..core import array as ip_array
from .utils import *
import numpy as np
from napari.layers.utils._link_layers import link_layers, unlink_layers

# TODO: add "edit" menu in napari

KEYS = {"hide_others": "Control-Shift-A",
        "link_selected_layers": "Control-G",
        "unlink_selected_layers": "Control-Shift-G",
        "layers_to_labels": "Alt-L",
        "crop": "Control-Shift-X",
        "to_front": "Control-Shift-F",
        "reset_view": "Control-Shift-R",
        "add_new_shape": "S",
        "proj": "Control-P",
        "duplicate_layer": "Control-Shift-D",
        }

# TODO: new images generated in napari does not inherite original metadata such as dirpath
# also, they may have float64

__all__ = list(KEYS.keys())

def bind_key(func):
    return napari.Viewer.bind_key(KEYS[func.__name__])(func)

@bind_key
def hide_others(viewer):
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
def link_selected_layers(viewer):
    """
    Link selected layers.
    """
    link_layers(viewer.layers.selection)
    
@bind_key
def unlink_selected_layers(viewer):
    """
    Unlink selected layers.
    """
    unlink_layers(viewer.layers.selection)

@bind_key
def to_front(viewer):
    """
    Let selected layers move to front.
    """
    not_selected_index = [i for i, l in enumerate(viewer.layers) 
                          if l not in viewer.layers.selection]
    viewer.layers.move_multiple(not_selected_index, 0)
    
@bind_key
def reset_view(viewer):
    """
    Reset translate/scale parameters to the initial value.
    """    
    for layer in viewer.layers.selection:
        # layer.translte[:] = 0. did not work
        layer.translate -= (layer.translate - layer.metadata["init_translate"])
        layer.scale = layer.metadata["init_scale"]

@bind_key
def layers_to_labels(viewer):
    """
    Convert manually drawn shapes to labels and store it.
    """        
    # determine destinations.
    destinations = [l.data for l in iter_selected_layer(viewer, "Image")]
    if len(destinations) == 0:
        destinations = [front_image(viewer).data]
    # TODO: image translation
    for dst in destinations:
        # check zoom_factors
        d = viewer.dims
        scale = {a: r[2] for a, r in zip(d.axis_labels, d.range)}
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
def crop(viewer):
    """
    Crop images with rectangle shapes.
    """        
    imglist = list(iter_selected_layer(viewer, "Image"))
    if len(imglist) == 0:
        imglist = [front_image(viewer)]
    
    rects = []
    for shape_layer in iter_selected_layer(viewer, "Shapes"):
        for shape, type_ in zip(shape_layer.data, shape_layer.shape_type):
            if type_ == "rectangle":
                rects.append(shape) # float pixel
                
    
    for rect in rects:
        if np.any(rect[0, -2:] == rect[1, -2:]):
            crop_func = crop_rectangle
        else:
            crop_func = crop_rotated_rectangle
        
        for layer in imglist:
            layer = viewer.add_layer(copy_layer(layer))
            dyx = layer.translate[-2:] / layer.scale[-2:]
            
            newdata, relative_translate = crop_func(layer.data, rect, dyx)
            if newdata.size <= 0:
                continue
            # TODO: not needed?
            if not isinstance(newdata, LabeledArray):
                scale = get_viewer_scale(viewer)
                axes = "".join(viewer.dims.axis_labels)
                newdata = ip_array(newdata, axes=axes)
                newdata.set_scale(**scale)
            layer.data = newdata
            translate = layer.translate
            translate[-2:] += relative_translate * layer.scale[-2:]
            layer.translate = translate
            layer.metadata.update({"init_translate": layer.translate, 
                                   "init_scale": layer.scale})
            
    # remove original images
    [viewer.layers.remove(img) for img in imglist]
    return None

@bind_key
def proj(viewer):
    """
    Projection
    """
    # TODO: Not all the parameters will be copied to new ones.
    layers = list(viewer.layers.selection)
    for layer in layers:
        data = layer.data
        kwargs = {}
        kwargs.update({"scale": layer.scale[-2:], 
                       "translate": layer.translate[-2:],
                       "blending": layer.blending,
                       "opacity": layer.opacity,
                       "name": layer.name+"-proj"})
        if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
            raise TypeError("Projection not supported.")
        elif isinstance(layer, napari.layers.Shapes):
            data = [d[:,-2:] for d in data]
            for k in ["face_color", "edge_color", "edge_width"]:
                kwargs[k] = getattr(layer, k)
            
            viewer.add_shapes(data, **kwargs)
        elif isinstance(layer, napari.layers.Points):
            data = data[:, -2:]
            for k in ["face_color", "edge_color", "size", "symbol"]:
                kwargs[k] = getattr(layer, k)
            kwargs["size"] = layer.size[:,-2:]
            
            viewer.add_points(data, **kwargs)
        elif isinstance(layer, napari.layers.Tracks):
            data = data[:, [0,-2,-1]]
            viewer.add_tracks(data, **kwargs)
        else:
            raise NotImplementedError(type(layer))
        
@bind_key
def duplicate_layer(viewer):
    """
    Duplicate selected layer(s).
    """
    [viewer.add_layer(copy_layer(layer)) for layer in list(viewer.layers.selection)]

def crop_rotated_rectangle(img, crds, dyx):
    # TODO: this does not work for memory maps
    crds = crds[:,-2:] - dyx
    cropped_img = img.rotated_crop(crds[1], crds[0], crds[2])
    translate = crds[0]
    return cropped_img, translate


def crop_rectangle(img, crds, dyx):
    crds = crds[:, -2:] - dyx
    start = crds[0, -2:]
    end = crds[2, -2:]
    sl = []
    for i in [0,1]:
        sl0 = sorted([start[i], end[i]])
        sl.append(slice(int(sl0[0])+1, int(sl0[1])+1))
    
    ndim = img.ndim
    area_to_crop = (slice(None),)*(ndim-2) + tuple(sl)
    
    translate = np.array([s.start for s in sl])
    cropped_img = img[area_to_crop]
    return cropped_img, translate