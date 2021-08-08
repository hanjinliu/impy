from __future__ import annotations
from ..utils.axesop import switch_slice
import os
import napari
import pandas as pd
import numpy as np
from inspect import signature
from dask import array as da
from skimage.measure import marching_cubes
import warnings

from .utils import *
from .mouse import *
from .widgets import TableWidget

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
# - Embed plot: https://github.com/napari/napari/blob/master/examples/mpl_plot.py

def change_theme(viewer):
    from napari.utils.theme import get_theme, register_theme
    theme = get_theme("dark")
    theme.update(console="rgb(20, 21, 22)",
                canvas="#090909")
    register_theme("night", theme)
    viewer.theme = "night"

class napariViewers:
    
    def __init__(self):
        self._viewers = {}
        self._front_viewer = None
    
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
    def viewer(self):
        """
        The most front viewer you're using
        """        
        return self._viewers[self._front_viewer]
        
    @property
    def layers(self):
        """
        Napari layer list.
        """        
        return self.viewer.layers
    
    @property
    def image_layers(self):
        """
        Return all the image layers as a list.
        """        
        return list(iter_layer(self.viewer, "Image"))
    
    @property
    def images(self):
        """
        Return all the images as a list.
        """        
        return list(layer_to_impy_object(self.viewer, a) 
                    for a in iter_layer(self.viewer, "Image"))
    
    @property
    def points(self):
        """
        Return all the images as a list.
        """        
        return list(layer_to_impy_object(self.viewer, a) 
                    for a in iter_layer(self.viewer, "Points"))
    
    @property
    def results(self):
        """
        Temporary results stored in the viewer.
        """        
        return self.viewer.window.results
    
    @property
    def selection(self) -> list:
        """
        Return selected layers' data as impy objects.
        """        
        return [layer_to_impy_object(self.viewer, layer) 
                for layer in self.viewer.layers.selection]
    
    @property
    def axes(self) -> str:
        return "".join(self.viewer.dims.axis_labels)
    
    @property
    def scale(self) -> dict[str: float]:
        """
        Dimension scales of the current viewer.
        """        
        d = self.viewer.dims
        return {a: r[2] for a, r in zip(d.axis_labels, d.range)}
    
    
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
        change_theme(viewer)
        
        viewer.window.file_menu.addSeparator()
        default_viewer_settings(viewer)
        load_mouse_callbacks(viewer)
        viewer.window.function_menu = viewer.window.main_menu.addMenu("&Functions")
        load_widgets(viewer)
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        self._viewers[key] = viewer
        self._front_viewer = key

        return None
        
    def add(self, obj=None, title=None, **kwargs):
        """
        Add images, points, labels, tracks etc to viewer.

        Parameters
        ----------
        obj : Any
            Object to add.
        """
        if title is None:
            if self._front_viewer is None:
                title = "impy"
            else:
                title = self._front_viewer
                
        if title not in self._viewers.keys():
            title = self._name(title)
            self.start(title)
        self._front_viewer = title
        
        # Add image and its labels
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        
        # Add points
        elif isinstance(obj, MarkerFrame):
            self._add_points(obj, **kwargs)
        
        # Add labels
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        
        # Add tracks
        elif isinstance(obj, TrackFrame):
            self._add_tracks(obj, **kwargs)
        
        # Add path
        elif isinstance(obj, PathFrame):
            self._add_paths(obj, **kwargs)
        
        # Add a table
        elif isinstance(obj, (pd.DataFrame, PropArray)):
            self._add_properties(obj, **kwargs)
        
        # Add a lazy-loaded image
        elif isinstance(obj, LazyImgArray):
            if obj.gb > Const["MAX_GB"] and self.viewer.dims.ndisplay == 3:
                raise MemoryError("Cannot send large files while the viewer is 3D mode.")
            with Progress("Sending Dask arrays to napari"):
                self._add_dask(obj, **kwargs)
        
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
            self.add(img, title=title, **kwargs)                
            
        # Add many objects of same type
        elif isinstance(obj, DataList):
            [self.add(each, title=title, **kwargs) for each in obj]
        
        elif obj is None:
            pass
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
    
    def preview(self, path:str, downsample_factor=4, dims=None, title=None, **kwargs):
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
        title : str, optional
            Title of the new viewer.
        """        
        if title is None:
            if self._front_viewer is None:
                title = "impy"
            else:
                title = self._front_viewer
                
        if title not in self._viewers.keys():
            title = self._name(title)
            self.start(title)
        self._front_viewer = title
        
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
    
    def bind(self, func=None, key="F1", progress:bool=False):
        """
        Decorator that makes it easy to call custom function on the viewer. Every time "F1" is pushed, 
        ``func(self)`` or `func(self, self.ax)` will be called. Returned values will appeded to
        ``self.results`` if exists.

        Parameters
        ----------
        func : callable
            Function to be called when ``key`` is pushed. This function must accept ``func(self)``, or
            ``func(self, self.ax)`` if you want to plot something inside the function. A figure widget will
            be added to the viewer unless ``func`` takes only one argument.
        key : str, default is "F1"
            Key binding.
        progress : bool, default is False
            If True, progress will be shown in the console like ``ImgArray``.
        
        Examples
        --------
        1. Calculate mean intensity of images.
        
            >>> @ip.gui.bind
            >>> def measure(gui):
            >>>     return gui.images[0].mean()
        
        2. Plot line scan of 2D image.
        
            >>> @ip.gui.bind
            >>> def profile(gui, ax=None)
            >>>     img = gui.images[0]
            >>>     line = gui.layers[-1].data[-1] # must be line!
            >>>     with ip.SetConst("SHOW_PROGRESS", False):
            >>>         scan = img.reslice(line)
            >>>     ax.plot(scan)
            >>>     return None
            
        """        
        def wrapper(f):
            if not callable(f):
                raise TypeError("func must be callable.")
            
            nparams = len(signature(f).parameters)
            
            if nparams == 1:
                @self.viewer.bind_key(key, overwrite=True)
                def _(viewer):
                    with Progress(f.__name__, out="stdout" if progress else None):
                        out = f(self)
                    win = viewer.window
                    if hasattr(win, "results") and isinstance(win.results, list):
                        win.results.append(out)
                    else:
                        win.results = [out]
                    viewer.status = f"'{f.__name__}' returned {out}"
            
            elif nparams > 1:
                if not hasattr(self, "ax"):
                    self._add_figure()
                @self.viewer.bind_key(key, overwrite=True)
                def _(viewer):
                    self.ax.cla()
                    with Progress(f.__name__, out="stdout" if progress else None):
                        out = f(self, self.ax)
                    self.fig.canvas.draw()
                    self.fig.tight_layout()
                    win = viewer.window
                    if out is not None:
                        if hasattr(win, "results") and isinstance(win.results, list):
                            win.results.append(out)
                        else:
                            win.results = [out]
                    viewer.status = f"'{f.__name__}' returned {out}"
                
            return f
        
        if func is None:
            return wrapper
        else:
            return wrapper(func)
    
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
            self._add_labels(img.labels, name=name, metadata={"destination_image": img})
        return None
    
    def _add_dask(self, img:LazyImgArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
                    
        scale = make_world_scale(img)
        
        if "contrast_limits" not in kwargs.keys():
            # contrast limits should be determined quickly.
            leny, lenx = img.shape[-2:]
            sample = img.img[..., ::leny//min(10, leny), ::lenx//min(10, lenx)]
            kwargs["contrast_limits"] = [float(sample.min().compute()), 
                                         float(sample.max().compute())]

        name = "No-Name" if img.name is None else img.name

        if chn_ax is not None:
            name = [f"[Lazy][C{i}]{name}" for i in range(img.sizeof("c"))]
        else:
            name = ["[Lazy]" + name]

        layer = self.viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                                      name=name if len(name)>1 else name[0], **kwargs)
        self.viewer.scale_bar.unit = img.scale_unit
        new_axes = [a for a in img.axes if a != "c"]
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self.viewer.dims.axis_labels):
            self.viewer.dims.axis_labels = new_axes
        return layer
    
    def _add_points(self, points, **kwargs):
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
            kw = dict(size=3.2, face_color=[0,0,0,0], metadata=metadata,
                      edge_color=self.viewer.window.cmap())
            kw.update(kwargs)
            self.viewer.add_points(each.values, scale=scale, **kw)
            
        return None
    
    def _add_labels(self, labels:Label, opacity:float=0.3, name:str|list[str]=None, **kwargs):
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
            
        for lbl, name in zip(lbls, names):
            self.viewer.add_labels(lbl.value, opacity=opacity, scale=scale, name=name, **kwargs)
        return None

    def _add_tracks(self, track:TrackFrame, **kwargs):
        if "c" in track._axes:
            track_list = track.split("c")
        else:
            track_list = [track]
            
        scale = make_world_scale(track[[a for a in track._axes if a != Const["ID_AXIS"]]])
        for tr in track_list:
            metadata = {"axes": str(tr._axes), "scale": tr.scale}
            self.viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
        
        return None
    
    def _add_paths(self, paths:PathFrame, **kwargs):
        if "c" in paths._axes:
            path_list = paths.split("c")
        else:
            path_list = [paths]
            
        scale = make_world_scale(paths[[a for a in paths._axes if a != Const["ID_AXIS"]]])
        kw = {"edge_color":"lime", "edge_width":0.3, "shape_type":"path"}
        kw.update(kwargs)

        for path in path_list:
            metadata = {"axes": str(path._axes), "scale": path.scale}
            paths = [single_path.values for single_path in path.split(Const["ID_AXIS"])]
            self.viewer.add_shapes(paths, scale=scale, metadata=metadata, **kw)
        
        return None
    
    def _add_properties(self, prop:PropArray|DataDict|pd.DataFrame):
        if isinstance(prop, PropArray):
            df = prop.as_frame()
            df.rename(columns = {"f": "value"}, inplace=True)
            table = TableWidget(self.viewer, df, name=prop.propname)
            
        elif isinstance(prop, pd.DataFrame):
            table = TableWidget(self.viewer, prop)
        else:
            raise TypeError(f"`prop` cannot be {type(prop)}")
        
        self.viewer.window.add_dock_widget(table, area="right", name=table.name)
        
        return None

    def _add_figure(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvas
        self.fig = plt.figure()
        self.viewer.window.add_dock_widget(FigureCanvas(self.fig), 
                                            name="Plot",
                                            area="right",
                                            allowed_areas=["right"])
        self.ax = self.fig.add_subplot(111)
        
    def _name(self, name="impy"):
        i = 0
        existing = self._viewers.keys()
        while name in existing:
            name += f"-{i}"
            i += 1
        return name
    
    
def default_viewer_settings(viewer):
    viewer.scale_bar.visible = True
    viewer.scale_bar.ticks = False
    viewer.scale_bar.font_size = 8 * Const["FONT_SIZE_FACTOR"]
    viewer.text_overlay.visible = True
    viewer.axes.colored = False
    viewer.window.cmap = ColorCycle()
    return None

def load_mouse_callbacks(viewer):
    from . import mouse
    for f in mouse.mouse_drag_callbacks:
        viewer.mouse_drag_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_wheel_callbacks:
        viewer.mouse_wheel_callbacks.append(getattr(mouse, f))
    for f in mouse.mouse_move_callbacks:
        viewer.mouse_move_callbacks.append(getattr(mouse, f))

def load_widgets(viewer):
    from . import _widgets
    for f in _widgets.__all__:
        getattr(_widgets, f)(viewer)

