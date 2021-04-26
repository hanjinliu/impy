from impy.func import determine_range
import napari
from .imgarray import ImgArray
from .labeledarray import LabeledArray
from .label import Label
from .specials import *


"""
# marker
viewer.layers[-1].name="peaks"
a=viewer.layers[-1]
a.symbol="arrow"
a.size = 0.2

 
# labeling
viewer.add_labels(img.labels)

# tracking
https://napari.org/tutorials/applications/cell_tracking.html
"""

# TODO: read layers

class napariWindow:
    def __init__(self):
        self.viewer = None
    
    @property
    def layers(self):
        return self.viewer.layers
    
    @property
    def last_layer(self):
        return self.viewer.layers[-1]
        
    def start(self):
        self.viewer = napari.Viewer(title="napari from impy")
    
    def add(self, obj, **kwargs):
        if self.viewer is None:
            self.start()
        # TODO: sometimes axes are not connected
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        elif isinstance(obj, (MarkerArray, PropArray, MarkerFrame)):
            self._add_points(obj, **kwargs)
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        elif isinstance(obj, TrackFrame):
            self._add_tracks(obj, **kwargs)
        else:
            raise TypeError(f"Could not interpret type {type(obj)}")
    
    def _add_image(self, img:ImgArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
        vmax, vmin = determine_range(img.value)
        self.viewer.add_image(img,
                              channel_axis=chn_ax,
                              scale=[img.scale[a] for a in img.axes if a != "c"],
                              name=img.name,
                              contrast_limits=[vmin, vmax],
                              **kwargs)
        
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=f"Label of {img.name}",
                             scale=[img.labels.scale[a] for a in img.labels.axes if a != "c"])
        return None
    
    def _add_points(self, points, **kwargs):
        if isinstance(points, PropArray):
            points = points.melt().values
        elif isinstance(points, MarkerArray):
            points = points.value.T
        elif isinstance(points, MarkerFrame):
            points = points.get_coords().values
            
        kw = dict(size=3.2, face_color=[0,0,0,0], edge_color="red")
        kw.update(kwargs)
        self.viewer.add_points(points, **kw)
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
        track_list = track.split("c") if "c" in track.col_axes else [track]
        
        for tr in track_list:
            self.viewer.add_tracks(tr, **kwargs)
        
        return None

window = napariWindow()
