from __future__ import annotations
import os
from typing import Any, Callable, NewType
import types
import napari
import sys
import warnings
import inspect
import pandas as pd
import numpy as np
from dask import array as da
from skimage.measure import marching_cubes

from .utils import *
from .mouse import *
from .widgets import TableWidget, LoggerWidget, ResultStackView

from ..utils.axesop import switch_slice
from ..collections import *
from ..arrays import *
from ..frame import *
from ..core import array as ip_array, aslazy as ip_aslazy, imread as ip_imread, read_meta
from ..utils.utilcls import Progress
from ..axes import ScaleDict
from .._const import Const

# TODO: 
# - Layer does not remember the original data after c-split ... this will be solved after 
#   layer group is implemented in napari.
# - 3D viewing in old viewer -> new viewer responds. napari's bug.
# - channel axis will be dropped in the future: https://github.com/napari/napari/issues/3019

# 0.4.11 updates
# - layer list context menu
# - point mask
# - doubleclick.connect
# - shapes event
# - "replace custom signals to accept/reject"?
# - highlight widget?

ImpyObject = NewType("ImpyObject", Any)
GUIcanvas = "module://impy.viewer._plt"
ResultsWidgetName = "Results"
MainPlotName = "Main Plot"

def _change_theme(viewer:"napari.Viewer"):
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
    
    def __getitem__(self, key:str) -> napariViewers:
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
    def results(self) -> ResultStackView:
        """
        Temporary results stored in the viewer.
        """    
        self.viewer.window._dock_widgets[ResultsWidgetName].show()
        return self.viewer.window._results

    
    @property
    def selection(self) -> list[ImpyObject]:
        """
        Return selected layers' data as a list of impy objects.
        """        
        return [layer_to_impy_object(self.viewer, layer) 
                for layer in self.viewer.layers.selection]
    
    @property
    def cursor_pos(self) -> np.ndarray:
        """
        Return cursor position. Scale is considered.
        """        
        return np.array(self.viewer.cursor.position)/self.scale
    
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
        return ScaleDict({a: r[2] for a, r in zip(d.axis_labels, d.range)})
    
    @property
    def fig(self):
        """
        ``matplotlib.figure.Figure`` object bound to the viewer.
        """
        if not (hasattr(self, "_fig") and 
                MainPlotName in self.viewer.window._dock_widgets.keys()):
            self._add_figure()
        return self._fig
        
    @property
    def table(self) -> TableWidget:
        if not (hasattr(self, "_table") and
                self._table.name in self.viewer.window._dock_widgets.keys()):
            self.add_table()
        return self._table
    
    @property
    def log(self) -> LoggerWidget:
        try:
            return self._log
        except AttributeError:
            logger = LoggerWidget(self.viewer)
            self.viewer.window.add_dock_widget(logger, name="Log", area="right", 
                                               allowed_areas=["right"])
            self._log = logger
            return self._log
    
    @property
    def params(self) -> dict:
        from magicgui.widgets import Label
        if hasattr(self, "_container"):
            kwargs = {wid.name: wid.value for wid in self._container if not isinstance(wid, Label)}
        else:
            kwargs = {}
            
        return kwargs
    
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
        viewer.window.layer_menu = viewer.window.main_menu.addMenu("&Layers")
        viewer.window.function_menu = viewer.window.main_menu.addMenu("&Functions")
        _add_results_widget(viewer)
        _load_widgets(viewer)
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        self._viewers[key] = viewer
        self._front_viewer = key

        return None

    def get(self, kind:str="image", layer_state:str="visible", returns:str="last") -> ImpyObject|list[ImpyObject]:
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
    
    def cursor_to_pixel(self, ref:"napari.layers.Image"|int|str|LabeledArray|LazyImgArray, 
                        ndim:int=None) -> np.ndarray:
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
        if isinstance(ref, (int, str)):
            layer = self.viewer.layers[ref]
        elif isinstance(ref, (LabeledArray, LazyImgArray)):
            for l in self.viewer.layers:
                if l.data is ref:
                    layer = l
                    break
            else:
                raise ValueError("Input image was not found in napari layer list.")
        
        elif isinstance(ref, (napari.layers.Image, napari.layers.Labels)):
            layer = ref
        else:
            raise TypeError("`layer` must be an image layer, int, str or impy's LabeledArray, "
                           f"but got {type(ref)}")
        
        if not isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
            raise TypeError(f"Layer {layer} is not an image or labels layer.")

        ndim = layer.data.ndim if ndim is None else ndim
        cursor_coords = np.array(self.viewer.cursor.position[-ndim:])
        pos = (cursor_coords - layer.translate)/layer.scale
        return (pos + 0.5).astype(np.int64)
        
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
            add_dask(self.viewer, obj, **kwargs)
        
        # Add an array as an image
        elif type(obj) is np.ndarray:
            self._add_image(ip_array(obj))
        
        # Add an dask array as an image
        elif type(obj) is da.core.Array:
            self._add_image(ip_aslazy(obj))
        
        # Add an image from a path
        elif isinstance(obj, str):
            viewer_imread(self.viewer, obj)
            
        # Add many objects of same type
        elif isinstance(obj, DataList):
            [self.add(each, **kwargs) for each in obj]
        
        elif obj is None:
            pass
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
    
    def preview(self, path:str, downsample_factor:int=4, dims:str|int=None, **kwargs):
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
             allowed_dims:int|tuple[int, ...]=(1, 2, 3)):
        """
        Decorator that makes it easy to call custom function on the viewer. Every time "F1" is pushed, 
        ``func(self, ...)`` will be called. Returned values will be appeded to ``self.results`` if exists.

        Parameters
        ----------
        func : callable
            Function to be called when ``key`` is pushed. This function must accept ``func(self, **kwargs)``.
            Docstring of this function will be displayed on the top of the parameter container as a tooltip.
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
        from ._plt import mpl
        from napari.utils.notifications import Notification, notification_manager
        
        if isinstance(allowed_dims, int):
            allowed_dims = (allowed_dims,)
        else:
            allowed_dims = tuple(allowed_dims)
                    
        def wrapper(f):
            if not callable(f):
                raise TypeError("func must be callable.")
                        
            source = inspect.getsource(f) # get source code
            params = inspect.signature(f).parameters
            gui_sym = list(params.keys())[0] # symbol of gui, func(gui, ...) -> "gui"
            
            _use_canvas = f"{gui_sym}.fig" in source or (use_plt and "plt" in source)
            _use_table = f"{gui_sym}.table" in source or f"{gui_sym}.register_point" in source or \
                f"{gui_sym}.register_shape" in source
            _use_log = f"{gui_sym}.log." in source or use_logger
            
            # show main plot widget if it is supposed to be used
            if _use_canvas:
                if not hasattr(self, "fig"):
                    self._add_figure()
                else:
                    self.viewer.window._dock_widgets[MainPlotName].show()
            
            _use_table and self.add_table()
            _use_log and self.log                
            
            self.add_parameter_container(f)
            
            @self.viewer.bind_key(key, overwrite=True)
            def _(viewer:"napari.Viewer"):
                if not viewer.dims.ndisplay in allowed_dims:
                    return None
                
                std_ = self if use_logger else None
                backend = mpl.get_backend()
                mpl.use(GUIcanvas)
                with Progress(f.__name__, out=None), setLogger(std_), mpl.style.context("night"):
                    try:
                        out = f(self, **self.params)
                    except Exception as e:
                        out = None
                        mpl.use(backend)
                        notification_manager.dispatch(Notification.from_exception(e))
                                            
                        
                if isinstance(out, types.GeneratorType):
                    # If original function returns a generator. This makes wrapper working almost same as
                    # napari's bind_key method.
                    yield from out
                
                elif out is not None:
                    self.results.append(out)
                
                if hasattr(self, "_fig"):
                    self.fig.tight_layout()
                    self.fig.canvas.draw()
                    
                mpl.use(backend)
                
                for layer in viewer.layers:
                    layer.refresh()
                return None
            
            return f
        
        return wrapper if func is None else wrapper(func)

    
    def bind_protocol(self, func=None, key1:str="F1", key2:str="F2", use_logger:bool=False, use_plt:bool=True, 
                      allowed_dims:int|tuple[int, ...]=(1, 2, 3), exit_with_error:bool=False):
        """
        Decorator that makes it easy to make protocol (series of function call) on the viewer. Unlike
        ``bind`` method, input function ``func`` must yield callable objects, from which parameter
        container will be generated. Simply, ``bind``-like method is called for every yielded function and
        each time parameter container will be renewed.
        Two keys, ``key1`` and ``key2``, are useful when you want to distinguish "repeat same function" and
        "proceed to next step". For instance, if in the first step you should add multiple points in the
        viewer as many as you want, it is hard for the protocol function to "know" whether you finish adding
        points. In this case, you can assign key "F1" to "add point" and key "F2" to "finish adding point"
        and inside the protocol function these different event can be distinguished by check whether 
        ``gui.proceed`` is True or not. See examples for details.
        
        Parameters
        ----------
        func : callable
            Protocol function. This function must accept ``func(self)`` and yield functions that accept
            ``f(self, **kwargs)`` . Docstring of the yielded functions will be displayed on the top of the 
            parameter container as a tooltip. Therefore it would be very useful if you write procedure of
            the protocol as docstrings. 
        key1 : str, default is "F1"
            First key binding. When this key is pushed, returned value will be appended to ``ip.gui.results``
            and attribute ``proceed`` will be False.
        key2 : str, default is "F2"
            Second key binding. When this key is pushed, returned value will be discarded and 
            attribute ``proceed`` will be True.
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
        exit_with_error :bool default is False
            If True, protocol will quit whenever exception is raised and key binding will be released. If
            False, protocol continues from the same step.
        
        Examples
        --------
        1. Draw a 3-D path and get the line scan.

            >>> @ip.gui.bind_protocol
            >>> def add_line(gui):
            >>>     def func(gui):
            >>>         '''
            >>>         Push F1 to add line.
            >>>         Push F2 to finish.
            >>>         '''
            >>>         return gui.cursor_pos
            >>>     while not gui.proceed:
            >>>         yield func
            >>>     line = np.stack(gui.results)
            >>>     img = gui.get("image")
            >>>     gui.register_shape(line, shape_type="path")
            >>>     plt.plot(img.reslice(line))
            >>>     plt.show()
        
        2. 3-D cropper.
        
            >>> @ip.gui.bind_protocol
            >>> def crop(gui):
            >>>     img = gui.get("image")
            >>>     layer = gui.viewer.add_shapes()
            >>>     layer.mode = "add_rectangle"
            >>>     def draw_rectangle(gui):
            >>>         '''draw a rectangle'''
            >>>         rect = gui.get("rectangle")[:,-2:]
            >>>         rect[:,0] /= img.scale["y"]
            >>>         rect[:,1] /= img.scale["x"]
            >>>         return rect.astype(np.int64)
            >>>     yield draw_rectangle
            >>>     def read_z(gui):
            >>>         '''set z borders'''
            >>>         return gui.stepof("z")
            >>>     yield read_z        # get one z-border coordinate
            >>>     yield read_z        # get the other z-border coordinate
            >>>     rect, z0, z1 = gui.results
            >>>     x0, _, _, x1 = sorted(rect[:, 1]) # get x-border coordinate
            >>>     y0, _, _, y1 = sorted(rect[:, 0]) # get y-border coordinate
            >>>     z0, z1 = sorted([z0, z1])
            >>>     gui.add(img[f"z={z0}:{z1+1};y={y0}:{y1+1};x={x0}:{x1+1}"])
        
        """        
        from ._plt import mpl
        from napari.utils.notifications import Notification, notification_manager
        
        allowed_dims = (allowed_dims,) if isinstance(allowed_dims, int) else tuple(allowed_dims)
                        
        def wrapper(protocol):
            if not callable(protocol):
                raise TypeError("func must be callable.")
                        
            source = inspect.getsource(protocol) # get source code
            params = inspect.signature(protocol).parameters
            gui_sym = list(params.keys())[0] # symbol of gui, func(gui, ...) -> "gui"
            
            _use_canvas = f"{gui_sym}.fig" in source or (use_plt and "plt" in source)
            _use_table = f"{gui_sym}.table" in source or f"{gui_sym}.register_point" in source or \
                f"{gui_sym}.register_shape" in source
            _use_log = f"{gui_sym}.log." in source or use_logger
            
            # show main plot widget if it is supposed to be used
            if _use_canvas:
                if not hasattr(self, "fig"):
                    self._add_figure()
                else:
                    self.viewer.window._dock_widgets[MainPlotName].show()
            
            _use_table and self.add_table()
            _use_log and self.log
            
            std_ = self if use_logger else None
            gen = protocol(self) # prepare generator from protocol function
            
            # initialize
            self.proceed = False
            self._yielded_func = next(gen)
            self.add_parameter_container(self._yielded_func)
            
            def _exit(viewer:"napari.Viewer"):
                # delete keymap
                viewer.keymap.pop(key1) 
                
                # delete temporal attribute
                del self.proceed
                del self._yielded_func 
                
                # delete widget
                viewer.window.remove_dock_widget(viewer.window._dock_widgets["Parameter Container"])
                return None
                
            @self.viewer.bind_key(key1, overwrite=True)
            def _1(viewer:"napari.Viewer"):
                return _base(viewer, proceed=False)
            
            @self.viewer.bind_key(key2, overwrite=True)
            def _2(viewer:"napari.Viewer"):
                return _base(viewer, proceed=True)
            
            def _base(viewer:"napari.Viewer", proceed=False):
                if not viewer.dims.ndisplay in allowed_dims:
                    return None
                backend = mpl.get_backend()
                mpl.use(GUIcanvas)
                with Progress(protocol.__name__, out=None), setLogger(std_), mpl.style.context("night"):
                    try:
                        self.proceed = proceed
                        out = self._yielded_func(self, **self.params)
                        if not proceed and out is not None:
                            self.results.append(out)
                            
                    except Exception as e:
                        notification_manager.dispatch(Notification.from_exception(e))
                        exit_with_error and _exit(viewer)
                            
                    else:
                        try:
                            next_func = next(gen)
                            if next_func != self._yielded_func:
                                # This avoid container renewing
                                self._yielded_func = next_func
                                self.add_parameter_container(self._yielded_func)
                
                        except StopIteration as e:
                            if e.value is not None:
                                self.results.append(e.value) # The last returned value is stored in e.value
                            _exit(viewer)
                            
                    finally:
                        if hasattr(self, "_fig"):
                            self.fig.tight_layout()
                            self.fig.canvas.draw()
                        mpl.use(backend)
                    
                for layer in viewer.layers:
                    layer.refresh()
                
                return None
            
            return protocol
        
        return wrapper if func is None else wrapper(func)
    
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
    
    def stepof(self, symbol:str) -> int:
        """
        Get the current step of certain axis.

        Parameters
        ----------
        symbol : str
            Axis symbol
        """        
        i = self.axes.find(symbol)
        return self.viewer.dims.current_step[i]

    def axisof(self, symbol:str) -> int:
        return self.axes.find(symbol)
    
    def register_point(self, data="cursor position", size:float=None, face_color=None, edge_color=None, 
                       properties:dict=None, **kwargs):
        """
        Register a point in a points layer, and link it to a table widget. Similar to "ROI Manager" in ImageJ.
        New points layer will be created when the first point is added.

        Parameters
        ----------
        data : array, optional
            Point coordinate. By default the cursor position willbe added.
        size : float, optional
            Point size.
        face_color : str or array, optional
            Face color of point.
        edge_color : str or array, optional
            Edge color of the point.
        properties : dict, optional
            Propertied of the point. Values in this parameter is also added to table.
        """        
        kwargs = dict(data=data, size=size, face_color=face_color, edge_color=edge_color, 
                      properties=properties, **kwargs)
        try:
            self.table.add_point(**kwargs)
        except Exception:
            self.add_table()
            self.table.add_point(**kwargs)
        return None

    def register_shape(self, data, shape_type="rectangle", face_color=None, edge_color=None, properties=None, 
                       **kwargs):
        """
        Register a shape in a shapes layer, and link it to a table widget. Similar to "ROI Manager" in ImageJ.
        New shapes layer will be created when the first shape is added.

        Parameters
        ----------
        data : array, optional
            Shape data.
        shape_type : str, default is "rectangle"
            Shape type of the new one.
        face_color : str or array, optional
            Face color of point.
        edge_color : str or array, optional
            Edge color of the point.
        properties : dict, optional
            Propertied of the point. Values in this parameter is also added to table.
        """        
        kwargs = dict(data=data, shape_type=shape_type, face_color=face_color, edge_color=edge_color, 
                      properties=properties, **kwargs)
        try:
            self.table.add_shape(**kwargs)
        except Exception:
            self.add_table()
            self.table.add_shape(**kwargs)
        return None
        
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
            data = data.as_frame()
            data.rename(columns = {"f": "value"}, inplace=True)
        
        self._table = add_table(self.viewer, data, columns, name)
        return self._table

    def use_logger(self) -> setLogger:
        """
        Return a context manager that all the texts will be printed in the logger.

        Examples
        --------
        
        >>> import logging
        >>> with ip.gui.use_logger():
        >>>     # both will be printed in the viewer's logger widget
        >>>     print("something")
        >>>     logging.warning("WARNING")
        """        
        return setLogger(self)
    
    def _add_figure(self):
        """
        Add figure canvas to the viewer.
        """        
        from ._plt import EventedCanvas, mpl, plt_figure
        # To block inline plot, we have to temporary change the backend.
        # It first seems that we can block inline plot with mpl.rc_context. Strangely it has no effect.
        # We have to call mpl.use to block it. 
        # See https://stackoverflow.com/questions/18717877/prevent-plot-from-showing-in-jupyter-notebook
        backend = mpl.get_backend()
        mpl.use("Agg")
        
        # To avoid irreversible backend change, we must ensure backend recovery by try/finally.
        try:
            self._fig = plt_figure()
            fig = self.viewer.window.add_dock_widget(EventedCanvas(self._fig), 
                                                     name=MainPlotName,
                                                     area="right",
                                                     allowed_areas=["right"])
            fig.setFloating(True)
            
        finally:
            mpl.use(backend)
        return None
    
    def add_parameter_container(self, f:Callable):
        """
        Make a parameter container widget from a function. Essentially same as magicgui's method.
        These parameters are accessible via ``ip.gui.param``.

        Parameters
        ----------
        f : Callable
            The function from which parameter types will be inferred.

        """        
        from magicgui.widgets import Container, create_widget, Label
        widget_name = "Parameter Container"
        
        params = inspect.signature(f).parameters
        
        if not f.__doc__ and len(params) == 1:
            return None
        
        if widget_name in self.viewer.window._dock_widgets:
            # Only with clear() method, the previous parameter labels are left on the widget.
            self._container.clear()
            while self._container.native.layout().count() > 0:
                self._container.native.layout().takeAt(0)
        else:
            self._container = Container(name=widget_name)
            wid = self.viewer.window.add_dock_widget(self._container, area="right", name=widget_name)
            wid.resize(140, 100)
            wid.setFloating(True)
        
        if f.__doc__:
            self._container.append(Label(value=f.__doc__))
            
        for i, (name, param) in enumerate(params.items()):
            if i == 0:
                continue
            value = None if param.default is inspect._empty else param.default
            
            widget = create_widget(value=value, annotation=param.annotation, 
                                   name=name, param_kind=param.kind)
            self._container.append(widget)
        
        self.viewer.window._dock_widgets[widget_name].show()
        
        return None
                
    
def _default_viewer_settings(viewer:"napari.Viewer"):
    viewer.scale_bar.visible = True
    viewer.scale_bar.ticks = False
    viewer.scale_bar.font_size = 8 * Const["FONT_SIZE_FACTOR"]
    viewer.text_overlay.visible = True
    viewer.axes.colored = False
    viewer.window.cmap = ColorCycle()
    return None

def _load_mouse_callbacks(viewer:"napari.Viewer"):
    from . import mouse
    for f in mouse.mouse_drag_callbacks:
        viewer.mouse_drag_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_wheel_callbacks:
        viewer.mouse_wheel_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_move_callbacks:
        viewer.mouse_move_callbacks.append(getattr(mouse, f))

def _load_widgets(viewer:"napari.Viewer"):
    from . import menus
    for f in menus.__all__:
        getattr(menus, f)(viewer)

def _add_results_widget(viewer:"napari.Viewer"):
    results = ResultStackView(viewer)
    viewer.window._results = results
    dock = viewer.window.add_dock_widget(results, name=ResultsWidgetName, area="right")
    dock.hide()
    return results

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
