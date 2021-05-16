import matplotlib.pyplot as plt
from .labeledarray import LabeledArray
from .phasearray import PhaseArray
from .label import Label
from .specials import *


"""
# marker
viewer.layers[-1].name="peaks"
a=viewer.layers[-1]
a.symbol="arrow"
a.size = 0.2
"""

# TODO: 
# - read layers
# - different name for different window?

class napariWindow:
    point_cmap = plt.get_cmap("rainbow", 16)
    
    def __init__(self):
        self.viewer = None
        self.point_color_id = 0
    
    @property
    def layers(self):
        return self.viewer.layers
    
    @property
    def last_layer(self):
        return self.viewer.layers[-1]
        
    def start(self):
        import napari
        self.viewer = napari.Viewer(title="impy")
    
    def add(self, obj, **kwargs):
        if self.viewer is None:
            self.start()
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
    
    def _add_image(self, img:LabeledArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
        if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
            kwargs["colormap"] = "hsv"
        self.viewer.add_image(img,
                              channel_axis=chn_ax,
                              scale=[img.scale[a] for a in img.axes if a != "c"],
                              name=img.name,
                              **kwargs)
        
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=f"Label of {img.name}",
                             scale=[img.labels.scale[a] for a in img.labels.axes if a != "c"])
        return None
    
    def _add_points(self, points, **kwargs):
        if isinstance(points, MarkerFrame):
            scale = [points.scale[a] for a in points._axes if a != "c"]
            points = points.get_coords()
        else:
            scale=None
        
        cmap = self.__class__.point_cmap
        if "c" in points._axes:
            pnts = points.split("c")
        else:
            pnts = [points]
            
        for each in pnts:
            kw = dict(size=3.2, face_color=[0,0,0,0], 
                      edge_color=list(cmap(self.point_color_id * (cmap.N//2+1) % cmap.N)))
            kw.update(kwargs)
            self.viewer.add_points(each.values, scale=scale, **kw)
            self.point_color_id += 1
        return None
    
    def _add_labels(self, labels:Label, opacity=0.3, **kwargs):
        if "c" in labels.axes:
            lbls = labels.split("c")
        else:
            lbls = [labels]
            
        for lbl in lbls:
            self.viewer.add_labels(lbl, opacity=opacity, **kwargs)
        return None

    def _add_tracks(self, track:TrackFrame, **kwargs):
        if "c" in track.col_axes:
            track_list = track.split("c")
        else:
            track_list = [track]
            
        scale = [track.scale[a] for a in track._axes if a not in "pc"]
        for tr in track_list:
            self.viewer.add_tracks(tr, scale=scale, **kwargs)
        
        return None
            

window = napariWindow()
