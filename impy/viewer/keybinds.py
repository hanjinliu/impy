from .utils import *
import napari
import numpy as np

@napari.Viewer.bind_key("Control-Shift-a")
def hide_others(viewer):
    """
    Make selected layers visible and others invisible. If key is pushed for a long time, then the visibility
    is restored upon release

    Parameters
    ----------
    viewer : napari.Viewer, optional
        Target viewer.
    """
    visibility = []
    selected = viewer.layers.selection
    for layer in viewer.layers:
        visibility.append(layer.visible)
        if layer in selected:
            layer.visible = True
        else:
            layer.visible = False
    

@napari.Viewer.bind_key("Control-h")
def hello(viewer):
    viewer.status = "Hello world"
    yield
    viewer.status = "goodbye world"

@napari.Viewer.bind_key("Alt-l")
def shapes_to_labels(viewer):
    """
    Convert manually drawn shapes to labels and store in `destination`.

    Parameters
    ----------
    destination : (list of) LabeledArray, optional
        To which labels will be stored, by default None
    viewer : napari.Viewer, optional
        Target viewer.
        
    Returns
    -------
    Label
        Last appended label.
    """        
    
    # determine
    destinations = [l.data for l in iter_selected_layer(viewer, "image")]
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
        # make labels from selected shapes
        shapes = [to_labels(layer, dst.shape, zoom_factor=zoom_factor) 
                    for layer in iter_selected_layer(viewer, "shape")]
        # append labels to each destination
        label = sum(shapes)
        if hasattr(dst, "labels"):
            print(f"Label already exist in {dst}. Overlapped.")
            del dst.labels
        dst.append_label(label)
        
    return dst.labels

@napari.Viewer.bind_key("Control-Shift-x")
def crop(viewer):
    """
    Crop images at the edges of the napari viewer. This function can be called with key binding by
    default.

    Parameters
    ----------
    dims : str, optional
        Which axes will not be cropped. Generally when an image stack is cropped at the edges, all 
        the images along t-axis should be cropped at the same edges. On the other hand, images at
        different positions (different p-coordinates) should not. That is why default is "tzc".
    viewer : napari.Viewer, optional
        Target viewer.
    """        
    
    imglist = list(filter(lambda x: isinstance(x, napari.layers.Image), viewer.layers.selection))
    count = 0
    for layer in imglist:
        sl = []
        translate = []
        for i, (start, end) in enumerate(layer.corner_pixels.T):
            start, end = int(start), int(end+1)
            if start+1 < end:
                if layer.data.axes[i] in "yx":
                    translate.append(start*layer.data.scale[layer.data.axes[i]])
                sl.append(slice(start, end))
            else:
                if layer.data.axes[i] in "tzc":
                    sl.append(slice(None))
                else:
                    sl.append(start)
        
        img = layer.data[tuple(sl)]
        if img.size > 0:
            kwargs = dict(name=layer.name+"-crop", 
                          colormap=layer.colormap,
                          contrast_limits=layer.contrast_limits,
                          translate=translate)
            
            scale = make_world_scale(img)
                        
            viewer.add_image(img, scale=scale, **kwargs)
            count += 1
    
    if count == 0:
        viewer.status = "No image was cropped"
    return None