from __future__ import annotations
import os
from typing import Any, NewType
import types
import napari
import sys
import pandas as pd
import numpy as np
import inspect
from dask import array as da
from skimage.measure import marching_cubes
import warnings

from .utils import *
from .mouse import *
from .widgets import TableWidget, LoggerWidget

from ..utils.axesop import switch_slice
from ..collections import *
from ..arrays import *
from ..frame import *
from ..core import array as ip_array, aslazy as ip_aslazy, imread as ip_imread, lazy_imread as ip_lazy_imread, read_meta
from ..utils.utilcls import Progress
from .._const import Const

# TODO: 
# - Layer does not remember the original data after c-split ... this will be solved after 
#   layer group is implemented in napari.
# - 3D viewing in old viewer -> new viewer responds. napari's bug.
# - channel axis will be dropped in the future: https://github.com/napari/napari/issues/3019
# - area=None for new dock widgets

ImpyObject = NewType("ImpyObject", Any)
GUIcanvas = "module://impy.viewer._plt"

def _change_theme(viewer):
    from napari.utils.theme import get_theme, register_theme, available_themes
    if "night" not in available_themes():
        theme = get_theme("dark")
        theme.update(console="rgb(20, 21, 22)",
                    canvas="#0F0F0F")
        register_theme("night", theme)
    viewer.theme = "night"

class napariViewers:
    """
    The controller of ``napari.Viewer``s from ``impy``. Always access by ``ip.gui``.
    """    
    def __init__(self):
        self._viewers:dict[str:"napari.Viewer"] = {}
        self._front_viewer:str = None
    
    def __repr__(self):
        w = "".join([f"<{k}>" for k in self._viewers.keys()])
        return f"{self.__class__}{w}"
    
    def __getitem__(self, key):
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
        """
        The most front viewer you're using
        """
        if self._front_viewer not in self._viewers.keys():
            self.start()
        return self._viewers[self._front_viewer]
        
    @property
    def layers(self) -> "napari.components.LayerList":
        """
        Napari layer list. Identical to ``ip.gui.viewer.layers``.
        """        
        return self.viewer.layers
    
    @property
    def current_slice(self) -> tuple[slice|int, ...]:
        """
        Return a tuple of slicer that corresponds to current field of view. For instance,
        when the viewer is displaying yx-plane at t=1, then this property returns 
        ``(1, slice(None), slice(None))``.
        """        
        current_step = list(self.viewer.dims.current_step)
        ndim = min(self.viewer.dims.ndisplay, self.viewer.dims.ndim)
        active_plane = list(self.viewer.dims.order[-ndim:])
        for i in active_plane:
            current_step[i] = slice(None)
        return tuple(current_step)
    
    @property
    def results(self) -> Any:
        """
        Temporary results stored in the viewer.
        """        
        try:
            return self.viewer.window.results
        except AttributeError:
            raise AttributeError("Viewer does not have temporary result.")
    
    @property
    def selection(self) -> list[ImpyObject]:
        """
        Return selected layers' data as a list of impy objects.
        """        
        return [layer_to_impy_object(self.viewer, layer) 
                for layer in self.viewer.layers.selection]
    
    @property
    def axes(self) -> str:
        """
        Axes information of current viewer. Defined to make compatible with ``ImgArray``.
        """        
        return "".join(self.viewer.dims.axis_labels)
    
    @property
    def scale(self) -> dict[str: float]:
        """
        Scale information of current viewer. Defined to make compatible with ``ImgArray``.
        """        
        d = self.viewer.dims
        return {a: r[2] for a, r in zip(d.axis_labels, d.range)}
    
    @property
    def fig(self):
        """
        ``matplotlib.figure.Figure`` object bound to the viewer.
        """
        if not (hasattr(self, "_fig") and 
                "Main Plot" in self.viewer.window._dock_widgets.keys()):
            self._add_figure()
        return self._fig
        
    @property
    def table(self):
        try:
            return self._table
        except AttributeError:
            self.add_table()
            return self._table
    
    @property
    def log(self):
        try:
            return self._log
        except AttributeError:
            logger = LoggerWidget(self.viewer)
            self.viewer.window.add_dock_widget(logger, name="Log", area="right", 
                                               allowed_areas=["right"])
            self._log = logger
            return self._log
    
    def start(self, key:str="impy"):
        """
        Create a napari window with name ``key``.
        """        
        if not isinstance(key, str):
            raise TypeError("`key` must be str.")
        if key in self._viewers.keys():
            raise ValueError(f"Key {key} already exists.")
        
        # load keybindings
        if not self._viewers:
            from . import keybinds
        
        viewer = napari.Viewer(title=key)
        _change_theme(viewer)
        
        viewer.window.file_menu.addSeparator()
        _default_viewer_settings(viewer)
        _load_mouse_callbacks(viewer)
        viewer.window.function_menu = viewer.window.main_menu.addMenu("&Functions")
        _load_widgets(viewer)
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        self._viewers[key] = viewer
        self._front_viewer = key

        return None

    def get(self, kind:str="image", layer_state:str="any", returns:str="last") -> ImpyObject|list[ImpyObject]:
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
        if kind in ("Image", "Labels", "Points", "Shapes", "Tracks"):
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
        
        
    def add(self, obj:ImpyObject=None, **kwargs):
        """
        Add images, points, labels, tracks etc to viewer.

        Parameters
        ----------
        obj : ImpyObject
            Object to add.
        """
        
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
            with Progress("Sending Dask arrays to napari"):
                add_dask(self.viewer, obj, **kwargs)
        
        # Add an array as an image
        elif type(obj) is np.ndarray:
            self._add_image(ip_array(obj))
        
        # Add an dask array as an image
        elif type(obj) is da.core.Array:
            self._add_image(ip_aslazy(obj))
        
        # Add an image from a path
        elif isinstance(obj, str):
            if not os.path.exists(obj):
                raise FileNotFoundError(f"Path does not exists: {obj}")
            size = os.path.getsize(obj)/1e9
            if size < Const["MAX_GB"]:
                img = ip_imread(obj)
            else:
                img = ip_lazy_imread(obj)
            self.add(img, **kwargs)                
            
        # Add many objects of same type
        elif isinstance(obj, DataList):
            [self.add(each, **kwargs) for each in obj]
        
        elif obj is None:
            pass
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
    
    def preview(self, path:str, downsample_factor=4, dims=None, **kwargs):
        """
        Preview a large image with a strided image.

        Parameters
        ----------
        path : str
            Path to the image file.
        downsample_factor : int, default is 4
            Image value is sampled every ``downsample_factor`` pixels.
        dims : str or int, optional
            Axes along which values will be down-sampled.
        """        
        if dims is None:
            meta = read_meta(path)
            dims = "zyx" if "z" in meta["axes"] else "yx"
        elif isinstance(dims, int):
            dims = "zyx" if dims == 3 else "yx"

        key = ";".join(f"{a}={downsample_factor//2}::{downsample_factor}" for a in dims)
        img = ip_imread(path, key=key)
        img.set_scale({a: img.scale[a]*downsample_factor for a in dims})
        trs = switch_slice(dims, img.axes, ifin=downsample_factor/2-0.5, ifnot=0.0)
        trs = np.array(trs) * np.array(list(img.scale.values()))/downsample_factor
        add_labeledarray(self.viewer, img, name=f"[Prev]{img.name}", translate=trs, **kwargs)
            
        return None
            
        
    def add_surface(self, image3d:LabeledArray, level:float=None, step_size:int=1, mask=None, **kwargs):
        """
        Add a surface layer from a 3D image.

        Parameters
        ----------
        image3d : LabeledArray
            3D image from which surface will be generated
        level, step_size, mask : 
            Passed to ``skimage.measure.marching_cubes``
        """        
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
    
    def bind(self, func=None, key:str="F1", use_logger:bool=False, use_plt:bool=True, 
             allowed_dims:int|tuple[int, ...]=(1, 2, 3), refresh:bool=True):
        """
        Decorator that makes it easy to call custom function on the viewer. Every time "F1" is pushed, 
        ``func(self, ...)`` will be called. Returned values will appeded to ``self.results`` if exists.

        Parameters
        ----------
        func : callable
            Function to be called when ``key`` is pushed. This function must accept ``func(self)``, or
            ``func(self, self.ax)`` if you want to plot something inside the function. A figure widget will
            be added to the viewer unless ``func`` takes only one argument.
        key : str, default is "F1"
            Key binding.
        use_logger : bool, default is False
            If True, all the texts that are printed out will be displayed in the log widget. In detail,
            ``sys.stdout`` and `sys.stderr` are substituted to the log widget during function call.
        use_plt : bool, default is True
            If True, the backend of ``matplotlib`` is set to "impy.viewer._plt" during function call. All
            the plot functions such as ``plt.plot`` will update the figure canvas inside the viewer. If
            ``plt.figure`` or other figure generation function is not called, the figure canvas will not be
            refreshed so that new results will drawn over the old ones.
        allowed_dims : int or tuple of int, default is (1, 2, 3)
            Function will not be called if the number of displayed dimensions does not match it.
        refresh : bool, default is True
            If refresh all the layers for every function call. Layers should be refreshed if their data are
            changed by function call. Set ``False`` if slow.
        
        Examples
        --------
        1. Calculate mean intensity of images.
        
            >>> @ip.gui.bind
            >>> def measure(gui):
            >>>     return gui.get("image").mean()
        
        2. Plot line scan of 2D image.
        
            >>> @ip.gui.bind
            >>> def profile(gui)
            >>>     img = gui.get("image")
            >>>     line = gui.get("line")
            >>>     scan = img.reslice(line)
            >>>     plt.plot(scan)
            >>>     return None
            
        """        
        from ._plt import canvas_plot, mpl
        if isinstance(allowed_dims, int):
            allowed_dims = (allowed_dims,)
        else:
            allowed_dims = tuple(allowed_dims)
            
        self.viewer.window.results = []
            
        def wrapper(f):
            if not callable(f):
                raise TypeError("func must be callable.")
                        
            source = inspect.getsource(f) # get source code
            params = inspect.signature(f).parameters
            gui_sym = list(params.keys())[0] # symbol of gui, func(gui, ...) -> "gui"
            
            _use_canvas = f"{gui_sym}.fig" in source or (use_plt and "plt" in source)
            _use_table = f"{gui_sym}.table" in source
            _use_log = f"{gui_sym}.log." in source or use_logger
            
            self._add_parameter_container(params)
            
            # show main plot widget if it is supposed to be used
            if _use_canvas:
                if not hasattr(self, "fig"):
                    self._add_figure()
                else:
                    self.viewer.window._dock_widgets["Main Plot"].show()
            
            _use_table and self.add_table()
            _use_log and self.log                
        
            @self.viewer.bind_key(key, overwrite=True)
            def _(viewer:"napari.Viewer"):
                if not viewer.dims.ndisplay in allowed_dims:
                    return None
                
                kwargs = {wid.name: wid.value for wid in self._container}
                std_ = self if use_logger else None
                with Progress(f.__name__, out=None), setLogger(std_):
                    if use_plt:
                        backend = mpl.get_backend()
                        mpl.use(GUIcanvas)
                        try:
                            with canvas_plot():
                                out = f(self, **kwargs)
                        except Exception:
                            mpl.use(backend)
                            raise
                        else:
                            mpl.use(backend)
                    else:
                        out = f(self, **kwargs)
                if isinstance(out, types.GeneratorType):
                    # If original function returns a generator. This makes wrapper working almost same as
                    # napari's bind_key method.
                    yield from out
                
                if _use_canvas:
                    self.fig.tight_layout()
                    self.fig.canvas.draw()
                win = viewer.window
                
                if out is not None:
                    win.results.append(out)
                    
                viewer.status = f"'{f.__name__}' returned {out}"
                
                if refresh:
                    for layer in viewer.layers:
                        layer.refresh()
                return None
            
            return f
        
        if func is None:
            return wrapper
        else:
            return wrapper(func)
    
    def goto(self, **kwargs) -> tuple[int, ...]:
        """
        Change the current step of the viewer.

        Examples
        --------
        1. Go to t=3.
        
            >>> ip.gui.goto(t=3)
        
        2. Go to t=3 and z=12. 
        
            >>> ip.gui.goto(t=3, z=12)
            
        """        
        step = list(self.viewer.dims.current_step)
        for axis, ind in kwargs.items():
            i = self.axisof(axis)
            step[i] = ind
        
        step = tuple(step)
        self.viewer.dims.current_step = step
        return step

    def axisof(self, symbol:str) -> int:
        return self.axes.find(symbol)
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #    Others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        
    def _add_image(self, img:LabeledArray, **kwargs):
        layer = add_labeledarray(self.viewer, img, **kwargs)
        if isinstance(layer, list):
            name = [l.name for l in layer]
        else:
            name = layer.name
        if hasattr(img, "labels"):
            add_labels(self.viewer, img.labels, name=name, metadata={"destination_image": img})
        return None
    
    def add_table(self, data=None, columns=None, name=None) -> TableWidget:
        """
        Add table widget in the viewer.

        Parameters
        ----------
        data : array-like, optional
            Initial data to add in the table. If not given, an empty table will be made.
        columns : sequence, optional
            Column names of the table.
        name : str, optional
            Name of the table widget.

        Returns
        -------
        TabelWidget
            Pointer to the table widget.
        """        
        if isinstance(data, PropArray):
            df = data.as_frame()
            df.rename(columns = {"f": "value"}, inplace=True)
        self._table = add_table(self.viewer, data, columns, name)
        return self._table

    def use_logger(self):
        return setLogger(self)

    def _add_figure(self):
        """
        Add figure canvas to the viewer.
        """        
        from ._plt import plt, canvas_plot, EventedCanvas, mpl
        
        # It first seems that we can block inline plot with mpl.rc_context. Strangely it has no effect.
        # We have to call mpl.use to block it. 
        # See https://stackoverflow.com/questions/18717877/prevent-plot-from-showing-in-jupyter-notebook
        backend = mpl.get_backend()
        mpl.use("Agg")
        with canvas_plot():
            self._fig = plt.figure()
            self.viewer.window.add_dock_widget(EventedCanvas(self._fig), 
                                               name="Main Plot",
                                               area="right",
                                               allowed_areas=["right"])
        
        mpl.use(backend)
        return None
    
    def _add_parameter_container(self, params:dict[str: inspect.Parameter]):
        from magicgui.widgets import Container, create_widget
        widget_name = "Parameter Controller"
        self._container = Container(name=widget_name)
                
        if widget_name in self.viewer.window._dock_widgets:
            # Call clear() is faster but the previous parameter labels are left on the widget with
            # unknown reason. Create new widget for now.
            dock = self.viewer.window._dock_widgets[widget_name]
            self.viewer.window.remove_dock_widget(dock)
        
        if len(params) == 1:
            return None
        
        self.viewer.window.add_dock_widget(self._container, area="right", name=widget_name)
            
        for i, (name, param) in enumerate(params.items()):
            if i == 0:
                continue
            value = None if param.default is inspect._empty else param.default
            
            widget = create_widget(value=value, annotation=param.annotation, 
                                   name=name, param_kind=param.kind)
            self._container.append(widget)
        
        self.viewer.window._dock_widgets[widget_name].show()
        
        return None
                
    
def _default_viewer_settings(viewer):
    viewer.scale_bar.visible = True
    viewer.scale_bar.ticks = False
    viewer.scale_bar.font_size = 8 * Const["FONT_SIZE_FACTOR"]
    viewer.text_overlay.visible = True
    viewer.axes.colored = False
    viewer.window.cmap = ColorCycle()
    return None

def _load_mouse_callbacks(viewer):
    from . import mouse
    for f in mouse.mouse_drag_callbacks:
        viewer.mouse_drag_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_wheel_callbacks:
        viewer.mouse_wheel_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_move_callbacks:
        viewer.mouse_move_callbacks.append(getattr(mouse, f))

def _load_widgets(viewer):
    from . import _widgets
    for f in _widgets.__all__:
        getattr(_widgets, f)(viewer)


class setLogger:
    def __init__(self, gui=None):
        self.gui = gui

    def __enter__(self):
        if self.gui:
            sys.stdout = self.gui.log
            sys.stderr = self.gui.log

    def __exit__(self, *args):
        if self.gui:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
