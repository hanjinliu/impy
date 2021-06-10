from __future__ import annotations
from ..imgarray import ImgArray
from ..labeledarray import LabeledArray
from ..phasearray import PhaseArray
from ..label import Label
from ..specials import *
from .utils import *
from .mouse import *


# TODO: 
# - Layer does not remember the original data after c-split ... this will be solved after 
#   layer group is implemented in napari.

        
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
        return list(map(self.get_data, self.viewer.layers.selection))
    
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
        
        default_viewer_settings(viewer)
        load_mouse_callbacks(viewer)
        load_widgets(viewer)
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        self._viewers[key] = viewer
        self._front_viewer = key
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
        
    def add(self, obj, title=None, **kwargs):
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
        elif isinstance(obj, MarkerFrame):
            self._add_points(obj, **kwargs)
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        elif isinstance(obj, TrackFrame):
            self._add_tracks(obj, **kwargs)
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
            
        scale = make_world_scale(track[[a for a in track._axes if a != "p"]])
        for tr in track_list:
            metadata = {"axes": str(tr._axes), "scale": tr.scale}
            self.viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
        
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

def load_widgets(viewer):
    from . import widgets
    for f in widgets.__all__:
        getattr(widgets, f)(viewer)

