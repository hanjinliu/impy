from __future__ import annotations
from typing import Any, NewType, TYPE_CHECKING
from pathlib import Path
import napari
import warnings
from weakref import WeakValueDictionary
import numpy as np

from .utils import (
    layer_to_impy_object,
    add_dask, 
    add_labeledarray,
    add_labels,
    add_paths,
    add_points,
    add_tracks,
    add_rois,
    viewer_imread,
    make_world_scale,
)

from impy.collections import *
from impy.arrays import *
from impy.core import array as ip_array
from impy.lazy import asarray as ip_aslazy
from impy.axes import ScaleView, AxisLike, Axes, Axis
from impy._const import Const

if TYPE_CHECKING:
    from napari.components import LayerList
    import pandas as pd
    from numpy.typing import ArrayLike

# TODO: 
# - Layer does not remember the original data after c-split ... this will be solved after 
#   layer group is implemented in napari.
# - channel axis will be dropped in the future: https://github.com/napari/napari/issues/3019

ImpyObject = NewType("ImpyObject", Any)

class napariViewers:
    """
    The controller of ``napari.Viewer``s from ``impy``. Always access by ``ip.gui``.
    """    
    def __init__(self):
        self._viewers: WeakValueDictionary[str, "napari.Viewer"] = WeakValueDictionary()
        self._front_viewer: str = None
        self._axes: Axes = None
    
    def __repr__(self):
        w = "".join([f"<{k}>" for k in self._viewers.keys()])
        return f"{self.__class__}{w}"
    
    def __getitem__(self, key: str) -> napariViewers:
        """
        This method looks strange but intuitive because you can access the last viewer by
        >>> ip.gui.add(...)
        while turn to another by
        >>> ip.gui["X"].add(...)

        Parameters
        ----------
        key : str
            Viewer's title
        """        
        if key in self._viewers.keys():
            self._front_viewer = key
        else:
            self.start(key)
        return self
    
    @property
    def viewer(self) -> "napari.Viewer":
        """The most front viewer you're using"""
        if self._front_viewer not in self._viewers.keys():
            self.start()
        return self._viewers[self._front_viewer]
        
    @property
    def layers(self) -> "LayerList":
        """Napari layer list. Identical to ``ip.gui.viewer.layers``."""        
        return self.viewer.layers
    
    @property
    def current_slice(self) -> tuple[slice | int, ...]:
        """
        Return a tuple of slicer that corresponds to current field of view. 
        
        For instance, when the viewer is displaying yx-plane at t=1, then this property 
        returns ``(1, slice(None), slice(None))``.
        """        
        current_step = list(self.viewer.dims.current_step)
        ndim = min(self.viewer.dims.ndisplay, self.viewer.dims.ndim)
        active_plane = list(self.viewer.dims.order[-ndim:])
        for i in active_plane:
            current_step[i] = slice(None)
        return tuple(current_step)

    
    @property
    def selection(self) -> list[ImpyObject]:
        """Return selected layers' data as a list of impy objects."""        
        return [layer_to_impy_object(self.viewer, layer) 
                for layer in self.viewer.layers.selection]
    
    @property
    def cursor_pos(self) -> np.ndarray:
        """Return cursor position. Scale is considered."""
        return np.array(self.viewer.cursor.position) / self.scale
    
    @property
    def axes(self) -> Axes:
        """
        Axes information of current viewer. 
        
        Defined to make compatible with ``ImgArray``.
        """
        d = self.viewer.dims
        unit = self.viewer.scale_bar.unit
        axes: list[Axis] = []
        for a, r in zip(d.axis_labels, d.range):
            if a in ["z", "y", "x"]:
                axis = Axis(a, {"scale": r[2], "unit": unit})
            else:
                axis = Axis(a)
            axes.append(axis)
        self._axes = Axes(axes)  # because scale-view uses a weakref
        return self._axes
    
    @property
    def scale(self) -> ScaleView:
        """
        Scale information of current viewer.
        
        Defined to make compatible with ``ImgArray``.
        """        
        return self.axes.scale
    
    def start(self, key: str = "impy"):
        """Create a napari window with name ``key``."""
        if not isinstance(key, str):
            raise TypeError("`key` must be str.")
        if key in self._viewers.keys():
            raise ValueError(f"Key {key} already exists.")
        
        viewer = napari.Viewer(title=key)
        self._viewers[key] = viewer
        self._front_viewer = key
        return None

    def get(
        self, 
        kind: str = "image",
        layer_state: str = "visible",
        returns: str = "last"
    ) -> ImpyObject | list[ImpyObject]:
        """
        Simple way to get impy object from viewer.

        Parameters
        ----------
        kind : str, optional
            Kind of layers/shapes to return.
            
                - "image": Image layer.
                - "labels": Labels layer
                - "points": Points layer.
                - "shapes": Shapes layer.
                - "tracks": Tracks layer.
                - "vectors":  Vectors layer.
                - "surface": Surface layer.
                - "line": Line shapes in Shapes layer.
                - "rectangle": Rectangle shapes in Shapes layer.
                - "path": Path shapes in Shapes layer.
                - "polygon": Polygon shapes in Shapes layer.
                - "ellipse": Ellipse shapes in Shapes layer.
                
        layer_state : {"selected", "visible", "any"}, default is "any"
            How to filter layer list.
            
                - "selected": Only selected layers will be searched.
                - "visible": Only visible layers will be searched.
                - "any": All the layers will be searched.    
                
        returns : {"first", "last", "all"}
            What will be returned in case that there are multiple layers/shapes.
            
                - "first": Only the first object will be returned.
                - "last": Only the last object will be returned.
                - "all": All the objects will be returned as a list.
        
        Returns
        -------
        ImgArray, Label, MarkerFrame or TrackFrame, np.ndarray, or list of one of them.
            impy object(s) that satisfies the options.
        
        Examples
        --------
        1. Get the front image.
        
            >>> ip.gui.get("image")
        
        2. Get all the selected images as a list.
            
            >>> ip.gui.get("image", layer_state="selected", returns="all")
            
        3. Get all the lines from the front visible shapes layer.
            
            >>> ip.gui.get("line", layer_state="visible") 
            
        """        
        if layer_state == "selected":
            layer_list = list(self.viewer.layers.selection)
        elif layer_state == "visible":
            layer_list = [layer for layer in self.viewer.layers if layer.visible]
        elif layer_state == "any":
            layer_list = self.viewer.layers
        else:
            raise ValueError("`filter` must be 'selected', 'visible' or 'any'")
            
        kind = kind.capitalize()
        out = []
        if kind in ("Image", "Labels", "Points", "Shapes", "Tracks", "Vectors", "Surface"):
            layer_type = getattr(napari.layers, kind)
            
            for layer in layer_list:
                if isinstance(layer, layer_type):
                    out.append(layer_to_impy_object(self.viewer, layer))
            
        elif kind in ("Line", "Rectangle", "Path", "Polygon", "Ellipse"):
            layer_type = napari.layers.Shapes
            shape_type = kind.lower()
            for layer in layer_list:
                if not isinstance(layer, layer_type):
                    continue
                for s, t in zip(layer.data, layer.shape_type):
                    if t == shape_type:
                        out.append(s)
            
        else:
            raise TypeError(f"Cannot interpret type {kind}")
        
        try:
            if returns == "first":
                out = out[0]
            elif returns == "last":
                out = out[-1]
            elif returns != "all":
                raise ValueError("`returns` must be 'first', 'last' or 'all'")
            
        except IndexError:
            if layer_state != "none":
                msg = f"No {layer_state} {kind.lower()} found in the viewer layer list."
            else:
                msg = f"No {kind.lower()} found in the viewer layer list."
            raise IndexError(msg)
        
        return out
    
    def cursor_to_pixel(
        self, 
        ref: "napari.layers.Image" | int | str | LabeledArray | LazyImgArray, 
        ndim: int = None
    ) -> np.ndarray:
        """
        With cursor position and a layer as inputs, this function returns the cursor "pixel" coordinates on the given
        layer. This function is useful when you want to get such as pixel value at the cursor position.

        Parameters
        ----------
        ref : napari.layers.Image, int, str, LabeledArray or LazyImgArray
            Reference layer or its identifier. To determine the reference layer, this parameter is interpreted in 
            different ways depending on its type:

            - napari.layers.Image ... layer itself 
            - int ... the index of layer list
            - str ... the name of layer list
            - LabeledArray or LazyImgArray ... layer that has same object as data
        
        ndim : int, optional
            If specified, the last ndim coordinates will be returned.
        
        Returns
        -------
        np.ndarray
            1-D, int64 array of cursor position along each dimension.
        """
        from napari.layers import Image, Labels
        if isinstance(ref, (int, str)):
            layer = self.viewer.layers[ref]
        elif isinstance(ref, (LabeledArray, LazyImgArray)):
            for l in self.viewer.layers:
                if l.data is ref:
                    layer = l
                    break
            else:
                raise ValueError("Input image was not found in napari layer list.")
        
        elif isinstance(ref, (Image, Labels)):
            layer = ref
        else:
            raise TypeError("`layer` must be an image layer, int, str or impy's LabeledArray, "
                           f"but got {type(ref)}")
        
        if not isinstance(layer, (Image, Labels)):
            raise TypeError(f"Layer {layer} is not an image or labels layer.")

        ndim = layer.data.ndim if ndim is None else ndim
        cursor_coords = np.array(self.viewer.cursor.position[-ndim:])
        pos = (cursor_coords - layer.translate)/layer.scale
        return (pos + 0.5).astype(np.int64)
        
    def add(self, obj: ImpyObject, **kwargs):
        """
        Add images, points, labels, tracks etc to viewer.

        Parameters
        ----------
        obj : ImpyObject
            Object to add.
        """        
        import pandas as pd
        from dask import array as da
        from impy.frame import MarkerFrame, TrackFrame, PathFrame
        
        try:
            self.viewer.window.activate()
        except RuntimeError:
            # Restart viewer
            viewer = napari.Viewer(title=self._front_viewer)
            self._viewers[self._front_viewer] = viewer
        
        # Add image and its labels
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        
        # Add points
        elif isinstance(obj, MarkerFrame):
            add_points(self.viewer, obj, **kwargs)
        
        # Add labels
        elif isinstance(obj, Label):
            add_labels(self.viewer, obj, **kwargs)
        
        # Add tracks
        elif isinstance(obj, TrackFrame):
            add_tracks(self.viewer, obj, **kwargs)
        
        # Add path
        elif isinstance(obj, PathFrame):
            add_paths(self.viewer, obj, **kwargs)
        
        # Add a table
        elif isinstance(obj, (pd.DataFrame, PropArray)):
            self.add_table(obj, **kwargs)
        
        # Add a lazy-loaded image
        elif isinstance(obj, LazyImgArray):
            if obj.gb > Const["MAX_GB"] and self.viewer.dims.ndisplay == 3:
                raise MemoryError("Cannot send large files while the viewer is 3D mode.")
            add_dask(self.viewer, obj, **kwargs)
        
        # Add an array as an image
        elif type(obj) is np.ndarray:
            self._add_image(ip_array(obj))
        
        # Add an dask array as an image
        elif type(obj) is da.core.Array:
            self._add_image(ip_aslazy(obj))
        
        # Add an image from a path
        elif isinstance(obj, (str, Path)):
            viewer_imread(self.viewer, obj)
            
        # Add many objects of same type
        elif isinstance(obj, DataList):
            [self.add(each, **kwargs) for each in obj]
        
        # Add a RoiList object as a shapes
        elif type(obj).__name__ == "RoiList":
            add_rois(self.viewer, obj, **kwargs)
        
        # Add a Roi object as a shape
        elif hasattr(obj, "from_imagejroi"):
            add_rois(self.viewer, [obj], **kwargs)
        
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
    
        
    def add_surface(
        self,
        image3d: LabeledArray,
        level: float = None,
        step_size: int = 1,
        mask=None,
        **kwargs
    ):
        """
        Add a surface layer from a 3D image.

        Parameters
        ----------
        image3d : LabeledArray
            3D image from which surface will be generated
        level, step_size, mask : 
            Passed to ``skimage.measure.marching_cubes``
        """        
        from skimage.measure import marching_cubes
        verts, faces, _, values = marching_cubes(image3d, level=level, 
                                                 step_size=step_size, mask=mask)
        scale = make_world_scale(image3d)
        name = f"[Surf]{image3d.name}"
        kw = dict(name=name, colormap="magma", scale=scale)
        kw.update(kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.viewer.add_surface((verts, faces, values), **kw)
        return None
    
    
    def goto(self, **kwargs) -> tuple[int, ...]:
        """
        Change the current step of the viewer.

        Examples
        --------
        1. Go to t=3.
        
            >>> ip.gui.goto(t=3)
        
        2. Go to t=3 and last z. 
        
            >>> ip.gui.goto(t=3, z=-1)
            
        """        
        step = list(self.viewer.dims.current_step)
        for axis, ind in kwargs.items():
            i = self.axisof(axis)
            if ind < 0:
                ind = self.viewer.dims.nsteps[i] + ind # support minus indexing
            step[i] = min(max(int(ind), 0), self.viewer.dims.nsteps[i]-1) # between min/max
        
        self.viewer.dims.current_step = step
        return step
    
    def stepof(self, symbol: AxisLike) -> int:
        """
        Get the current step of certain axis.

        Parameters
        ----------
        symbol : AxisLike
            Axis symbol
        """        
        i = self.axes.find(symbol)
        return self.viewer.dims.current_step[i]

    def axisof(self, symbol: AxisLike) -> int:
        return self.axes.find(symbol)
    
    def _add_image(self, img: LabeledArray, **kwargs):
        layer = add_labeledarray(self.viewer, img, **kwargs)
        if isinstance(layer, list):
            name = [l.name for l in layer]
        else:
            name = layer.name
        if img.labels is not None:
            add_labels(self.viewer, img.labels, name=name, metadata={"destination_image": img})
        return None
    
    def add_table(self, table: pd.DataFrame | dict[str, ArrayLike], **kwargs):
        from magicgui.widgets import Table
        table = Table(value=table)
        self.viewer.window.add_dock_widget(table, **kwargs)
        return table
    
    def add_image(self, *args, **kwargs):
        return self.viewer.add_image(*args, **kwargs)
    
    def add_shapes(self, *args, **kwargs):
        return self.viewer.add_shapes(*args, **kwargs)
    
    def add_surface(self, *args, **kwargs):
        return self.viewer.add_surface(*args, **kwargs)
    
    def add_points(self, *args, **kwargs):
        return self.viewer.add_points(*args, **kwargs)
    
    def add_labels(self, *args, **kwargs):
        return self.viewer.add_labels(*args, **kwargs)
    
    def add_vectors(self, *args, **kwargs):
        return self.viewer.add_vectors(*args, **kwargs)
    
    def add_tracks(self, *args, **kwargs):
        return self.viewer.add_tracks(*args, **kwargs)
