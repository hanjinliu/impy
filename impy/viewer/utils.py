from ..specials import MarkerFrame, TrackFrame
from ..utilcls import ImportOnRequest
napari = ImportOnRequest("napari")

def get_data(layer):
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
    if isinstance(layer, (napari.layers.Image, napari.layers.Labels, napari.layers.Shapes)):
        return layer.data
    elif isinstance(layer, napari.layers.Points):
        df = MarkerFrame(layer.data, columns=layer.metadata["axes"])
        df.set_scale(layer.metadata["scale"])
        return df
    elif isinstance(layer, napari.layers.Tracks):
        df = TrackFrame(layer.data, columns=layer.metadata["axes"])
        df.set_scale(layer.metadata["scale"])
        return df
    else:
        raise NotImplementedError(type(layer))

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
    if layer_type == "shape":
        layer_type = napari.layers.Shapes
    elif layer_type == "image":
        layer_type = napari.layers.Image
    elif layer_type == "point":
        layer_type = napari.layers.Points
    else:
        raise NotImplementedError
    
    for layer in viewer.layers:
        if isinstance(layer, layer_type):
            yield layer

def iter_selected_layer(viewer, layer_type:str):
    if layer_type == "shape":
        layer_type = napari.layers.Shapes
    elif layer_type == "image":
        layer_type = napari.layers.Image
    elif layer_type == "point":
        layer_type = napari.layers.Points
    else:
        raise NotImplementedError
    
    for layer in viewer.layers.selection:
        if isinstance(layer, layer_type):
            yield layer

def front_image(viewer):
    """
    From list of image layers return the most front visible image.
    """        
    front = None
    for img in iter_layer(viewer, "image"):
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