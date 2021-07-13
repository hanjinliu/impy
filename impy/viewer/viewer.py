from __future__ import annotations
from ..datalist import DataList
import napari
import pandas as pd
from ..arrays import *
from ..frame import *
from ..core import array as ip_array
from .utils import *
from .mouse import *
from ..utilcls import ArrayDict, Progress
from .widgets import _make_table_widget


# TODO: 
# - Layer does not remember the original data after c-split ... this will be solved after 
#   layer group is implemented in napari.
# - 3D viewing in old viewer -> new viewer responds. napari's bug.
# - channel axis will be dropped in the future: https://github.com/napari/napari/issues/3019

        
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
        return self._viewers[self._front_viewer]
        
    @property
    def layers(self):
        return self.viewer.layers
    
    @property
    def selection(self) -> list:
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
    
    @property
    def front_image(self):
        return front_image(self.viewer)
    
    def start(self, key:str="impy"):
        """
        Create a napari window with name `key`.
        """        
        if not isinstance(key, str):
            raise TypeError("`key` must be str.")
        if key in self._viewers.keys():
            raise ValueError(f"Key {key} already exists.")
        
        # load keybindings
        if not self._viewers:
            from . import keybinds
        
        viewer = napari.Viewer(title=key)
        
        viewer.window.file_menu.addSeparator()
        default_viewer_settings(viewer)
        load_mouse_callbacks(viewer)
        load_widgets(viewer)
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        self._viewers[key] = viewer
        self._front_viewer = key
        viewer.window.n_table = 0
        return None
    
    def get_data(self, layer):
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
        return layer_to_impy_object(self.viewer, layer)
        
    def add(self, obj=None, title=None, **kwargs):
        """
        Add images, points, labels, tracks or graph to viewer.

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
            
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        elif isinstance(obj, DataList):
            [self.add(each, title=title, **kwargs) for each in obj]
        elif isinstance(obj, MarkerFrame):
            self._add_points(obj, **kwargs)
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        elif isinstance(obj, TrackFrame):
            self._add_tracks(obj, **kwargs)
        elif isinstance(obj, PathFrame):
            self._add_paths(obj, **kwargs)
        elif isinstance(obj, (pd.DataFrame, PropArray, ArrayDict)):
            self._add_properties(obj, **kwargs)
        elif isinstance(obj, LazyImgArray):
            with Progress("Sending Dask arrays to napari"):
                self._add_dask(obj, **kwargs)
        elif type(obj) is np.ndarray:
            self._add_image(ip_array(obj))
        elif obj is None:
            pass
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
                
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #    Others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def _iter_layer(self, layer_type:str):
        return iter_layer(self.viewer, layer_type)
    
    def _iter_selected_layer(self, layer_type:str):
        return iter_selected_layer(self.viewer, layer_type)
        
    def _add_image(self, img:LabeledArray, **kwargs):
        layer = add_labeledarray(self.viewer, img, **kwargs)
        if isinstance(layer, list):
            name = [l.name for l in layer]
        else:
            name = layer.name
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=name)
        return None
    
    def _add_dask(self, img:LazyImgArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
                    
        scale = make_world_scale(img)
        
        if "contrast_limits" not in kwargs.keys():
            # contrast limits should be determined quickly.
            leny, lenx = img.shape[-2:]
            sample = img.img[..., ::leny//3, ::lenx//3]
            kwargs["contrast_limits"] = [sample.min().compute(), 
                                         sample.max().compute()]
        
        name = "No-Name" if img.name is None else img.name
        if chn_ax is not None:
            name = [f"[Preview][C{i}]{name}" for i in range(img.sizeof("c"))]
        else:
            name = ["[Preview]" + name]
        layer = self.viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                                      name=name if len(name)>1 else name[0],
                                      **kwargs)
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
            self.viewer.add_labels(lbl, opacity=opacity, scale=scale, name=name, **kwargs)
        return None

    def _add_tracks(self, track:TrackFrame, **kwargs):
        if "c" in track._axes:
            track_list = track.split("c")
        else:
            track_list = [track]
            
        scale = make_world_scale(track[[a for a in track._axes if a != ID_AXIS]])
        for tr in track_list:
            metadata = {"axes": str(tr._axes), "scale": tr.scale}
            self.viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
        
        return None
    
    def _add_paths(self, paths:PathFrame, **kwargs):
        if "c" in paths._axes:
            path_list = paths.split("c")
        else:
            path_list = [paths]
            
        scale = make_world_scale(paths[[a for a in paths._axes if a != ID_AXIS]])
        kw = {"edge_color":"lime", "edge_width":0.3, "shape_type":"path"}
        kw.update(kwargs)

        for path in path_list:
            metadata = {"axes": str(path._axes), "scale": path.scale}
            paths = [single_path.values for single_path in path.split(ID_AXIS)]
            self.viewer.add_shapes(paths, scale=scale, metadata=metadata, **kw)
        
        return None
    
    def _add_properties(self, prop:PropArray|ArrayDict|pd.DataFrame):
        QtViewerDockWidget = napari._qt.widgets.qt_viewer_dock_widget.QtViewerDockWidget
        if isinstance(prop, PropArray):
            df = prop.as_frame()
            df.rename(columns = {"f": "value"}, inplace=True)
            table = _make_table_widget(df, name=prop.propname)
        elif isinstance(prop, ArrayDict):
            data = None
            for k, pr in prop.items():
                df = pr.as_frame()
                if data is None:
                    data = df.rename(columns = {"f": k})
                else:
                    data[k] = df["f"]
            table = _make_table_widget(data)
        elif isinstance(prop, pd.DataFrame):
            table = _make_table_widget(prop)
        else:
            raise TypeError(f"`prop` cannot be {type(prop)}")
                
        widget = QtViewerDockWidget(self.viewer.window.qt_viewer, table, name="Table",
                                    area="right", add_vertical_stretch=True)
        self.viewer.window._add_viewer_dock_widget(widget, tabify=self.viewer.window.n_table>0)
        self.viewer.window.n_table += 1
        return None

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
    viewer.scale_bar.font_size = 8
    viewer.text_overlay.visible = True
    viewer.axes.visible = True
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
    from . import widgets
    for f in widgets.__all__:
        getattr(widgets, f)(viewer)

