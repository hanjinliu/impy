from __future__ import annotations
import warnings
from typing import TYPE_CHECKING, TypeVar
import numpy as np
import napari
import os

from ..arrays import *
from .._const import Const
from ..core import imread, lazy_imread

if TYPE_CHECKING:
    from ..frame import TrackFrame, PathFrame, AxesFrame
    from napari.layers import Shapes


def iter_layer(viewer:"napari.Viewer", layer_type: str):
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

def iter_selected_layer(viewer: "napari.Viewer", layer_type:str | list[str]):
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

def to_labels(layer: "Shapes", labels_shape, zoom_factor=1):
    return layer._data_view.to_labels(
        labels_shape=labels_shape, zoom_factor=zoom_factor
    )

def make_world_scale(obj):
    scale = []
    for a in obj._axes:
        if a in ["z", "y", "x"]:
            scale.append(obj.scale[a])
        elif a == "c":
            pass
        else:
            scale.append(1)
    return scale

def viewer_imread(viewer: "napari.Viewer", path: str):    
    if "*" in path or os.path.getsize(path)/1e9 < Const["MAX_GB"]:
        img = imread(path)
    else:
        img = lazy_imread(path)
    layer = add_labeledarray(viewer, img)
    return layer

def add_labeledarray(viewer: "napari.Viewer", img: LabeledArray, **kwargs):
    if not img.axes.is_sorted() and img.ndim > 2:
        msg = (
            f"Input image has axes that are not correctly sorted: {img.axes}. "
            "This may cause unexpected results."
        )
        warnings.warn(msg, UserWarning)
    chn_ax = img.axisof("c") if "c" in img.axes else None
        
    if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
        kwargs["colormap"] = "hsv"
        kwargs["contrast_limits"] = img.border
    elif img.dtype.kind == "c" and not "colormap" in kwargs.keys():
        kwargs["colormap"] = "plasma"
    
    scale = make_world_scale(img)
    
    if "name" in kwargs:
        name = kwargs.pop("name")
    else:
        name = img.name
        if chn_ax is not None:
            name = [f"[C{i}]{name}" for i in range(img.shape.c)]
        else:
            name = [name]
    
    if img.dtype.kind == "c":
        input = ComplexArrayView(img)
    else:
        input = img
        
    layer = viewer.add_image(
        input,
        channel_axis=chn_ax,
        scale=scale, 
        name=name if len(name) > 1 else name[0],
        **kwargs
    )
    
    if viewer.scale_bar.unit:
        if viewer.scale_bar.unit != img.scale_unit:
            msg = (
                f"Incompatible scales. Viewer is {viewer.scale_bar.unit} while "
                f"image is {img.scale_unit}."
            )
            warnings.warn(msg)
    else:
        viewer.scale_bar.unit = img.scale_unit
        
    new_axes = [str(a) for a in img.axes if a != "c"]
    # add axis labels to slide bars and image orientation.
    if len(new_axes) >= len(viewer.dims.axis_labels):
        viewer.dims.axis_labels = new_axes
    return layer

def add_labels(
    viewer: "napari.Viewer",
    labels: Label,
    opacity: float = 0.3,
    name: str | list[str] | None = None, 
    **kwargs
):
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
        
    kw = dict(opacity=opacity, scale=scale)
    kw.update(kwargs)
    
    out_layers = []
    for lbl, name in zip(lbls, names):
        layer = viewer.add_labels(lbl.value, name=name, **kw)
        out_layers.append(layer)
    return out_layers

def add_dask(viewer: "napari.Viewer", img: LazyImgArray, **kwargs):
    chn_ax = img.axisof("c") if "c" in img.axes else None
                
    scale = make_world_scale(img)
    name = img.name

    if chn_ax is not None:
        name = [f"[Lazy][C{i}]{name}" for i in range(img.shape.c)]
    else:
        name = ["[Lazy]" + name]

    layer = viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                             name=name if len(name)>1 else name[0], **kwargs)
    viewer.scale_bar.unit = img.scale_unit
    new_axes = [str(a) for a in img.axes if a != "c"]
    # add axis labels to slide bars and image orientation.
    if len(new_axes) >= len(viewer.dims.axis_labels):
        viewer.dims.axis_labels = new_axes
    return layer

def add_points(viewer:"napari.Viewer", points: AxesFrame, **kwargs):
    from ..frame import MarkerFrame
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
        kw = dict(
            size=3.2,
            face_color=[0, 0, 0, 0],
            metadata=metadata,
        )
        kw.update(kwargs)
        viewer.add_points(each.values, scale=scale, **kw)
        
    return None

def add_tracks(viewer:"napari.Viewer", track: "TrackFrame", **kwargs):
    if "c" in track._axes:
        track_list = track.split("c")
    else:
        track_list = [track]
        
    scale = make_world_scale(
        track[[a for a in track._axes if a != Const["ID_AXIS"]]]
    )
    for tr in track_list:
        metadata = {"axes": str(tr._axes), "scale": tr.scale}
        viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
    
    return None

def add_paths(viewer: "napari.Viewer", paths: PathFrame, **kwargs):
    if "c" in paths._axes:
        path_list = paths.split("c")
    else:
        path_list = [paths]
        
    scale = make_world_scale(
        paths[[a for a in paths._axes if a != Const["ID_AXIS"]]]
    )
    kw = {"edge_color": "lime", "edge_width": 0.3, "shape_type": "path"}
    kw.update(kwargs)

    for path in path_list:
        metadata = {"axes": str(path._axes), "scale": path.scale}
        paths = [single_path.values for single_path in path.split(Const["ID_AXIS"])]
        viewer.add_shapes(paths, scale=scale, metadata=metadata, **kw)
    
    return None

def get_viewer_scale(viewer: "napari.Viewer"):
    return {a: r[2] for a, r in zip(viewer.dims.axis_labels, viewer.dims.range)}

def layer_to_impy_object(viewer: "napari.Viewer", layer):
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
    from napari.layers import Image, Labels, Shapes, Points, Tracks
    if isinstance(layer, (Image, Labels)):
        # manually drawn ones are np.ndarray, need conversion
        if type(data) is np.ndarray:
            ndim = data.ndim
            axes = axes[-ndim:]
            if isinstance(layer, Image):
                data = ImgArray(data, name=layer.name, axes=axes, dtype=layer.data.dtype)
            else:
                try:
                    data = layer.metadata["destination_image"].labels
                except (KeyError, AttributeError):
                    data = Label(data, name=layer.name, axes=axes)
            data.set_scale({k: v for k, v in scale.items() if k in axes})
        return data
    elif isinstance(layer, Shapes):
        return data
    elif isinstance(layer, Points):
        from ..frame import MarkerFrame
        ndim = data.shape[1]
        axes = axes[-ndim:]
        df = MarkerFrame(data, columns=layer.metadata.get("axes", axes))
        df.set_scale(layer.metadata.get("scale", 
                                        {k: v for k, v in scale.items() if k in axes}))
        return df.as_standard_type()
    elif isinstance(layer, Tracks):
        from ..frame import TrackFrame
        ndim = data.shape[1]
        axes = axes[-ndim:]
        df = TrackFrame(data, columns=layer.metadata.get("axes", axes))
        df.set_scale(layer.metadata.get("scale", 
                                        {k: v for k, v in scale.items() if k in axes}))
        return df.as_standard_type()
    else:
        raise NotImplementedError(type(layer))

def get_a_selected_layer(viewer:"napari.Viewer"):
    selected = list(viewer.layers.selection)
    if len(selected) == 0:
        raise ValueError("No layer is selected.")
    elif len(selected) > 1:
        raise ValueError("More than one layers are selected.")
    return selected[0]


_A = TypeVar("_A", bound=np.ndarray)

class ComplexArrayView:
    """View a complex array in napari."""
    def __init__(self, data: _A):
        if data.dtype.kind != "c":
            raise TypeError("Input was not a complex array.")
        self._data = data
    
    def __array__(self, dtype=None) -> np.ndarray:
        return np.abs(self._data)
    
    @property
    def data_raw(self) -> _A:
        """Return the raw data."""
        return self._data
    
    @property
    def shape(self) -> tuple:
        """Shape of array."""
        return self._data.shape
    
    @property
    def ndim(self) -> int:
        """Number of dimensions of array."""
        return self._data.ndim
    
    @property
    def dtype(self) -> np.dtype:
        """Data type of array."""
        return self._data.dtype
    
    @property
    def nbytes(self) -> int:
        """Bytes of array."""
        return self._data.nbytes
    
    def __getitem__(self, key):
        return self.__class__(self._data[key])