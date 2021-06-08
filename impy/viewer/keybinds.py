from .utils import *
import numpy as np

KEYS = {"hide_others": "Control-Shift-A",
        "layers_to_labels": "Alt-L",
        "crop": "Control-Shift-X",
        "to_front": "Control-Shift-F"}


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
def to_front(viewer):
    """
    Let selected layers move to front.
    """
    not_selected_index = [i for i, l in enumerate(viewer.layers) 
                          if l not in viewer.layers.selection]
    viewer.layers.move_multiple(not_selected_index, 0)
    
    

@bind_key
def layers_to_labels(viewer):
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
    Crop images at the edges of the napari viewer. This function can be called with key binding by
    default.
    """        
    imglist = list(iter_selected_layer(viewer, "Image"))
    if len(imglist) == 0:
        imglist = [front_image(viewer)]
    count = 0
    for layer in imglist:
        sl = []
        translate = []
        for i, (start, end) in enumerate(layer.corner_pixels.T):
            start, end = int(start), int(end+1)
            if layer.data.axes[i] in "tzc":
                sl.append(slice(None))
                translate.append(0.0)
            elif layer.data.axes[i] in "yx":
                if start+1 < end:
                    sl.append(slice(start, end))
                else:
                    viewer.status = "Failed to crop."
                    return None
                translate.append(start*layer.data.scale[layer.data.axes[i]] + layer.translate[i])
            else:
                translate.append(0.0)
                sl.append(start)
        
        img = layer.data[tuple(sl)]
        if img.size > 0:
            kwargs = dict(name=layer.name+"-crop", 
                          colormap=layer.colormap,
                          contrast_limits=layer.contrast_limits,
                          translate=translate)
            
            scale = make_world_scale(img)
                        
            layer = viewer.add_image(img, scale=scale, **kwargs)
            count += 1
    
    if count == 0:
        viewer.status = "No image was cropped"
    return None

def project_labels(viewer):
    pass