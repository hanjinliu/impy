import napari
from .imgarray import ImgArray
from .labeledarray import LabeledArray
from .label import Label
from .specials import MarkerArray, PropArray
from .func import complement_axes, del_axis


"""
viewer = napari.Viewer()
viewer.add_image(img, name="raw image")
viewer.add_image(img2, name="DOG filter")

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
            
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        elif isinstance(obj, (MarkerArray,PropArray)):
            self._add_points(obj, **kwargs)
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        else:
            raise TypeError(f"Could not interpret type {type(obj)}")
    
    def _add_image(self, img:ImgArray, **kwargs):
        if "c" in img.axes:
            self.viewer.add_image(img,
                                  channel_axis=img.axisof("c"),
                                  scale=[img.scale[a] for a in img.axes],
                                  name=img.name, 
                                  **kwargs)
        else:
            self.viewer.add_image(img,
                                  scale=[img.scale[a] for a in img.axes],
                                  name=img.name, 
                                  **kwargs)
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=f"Label of {img.name}")
        return None
    
    def _add_points(self, points:MarkerArray, size=1.5, face_color="red", edge_color=None, **kwargs):
        if isinstance(points, PropArray):
            points = points.melt()
        self.viewer.add_points(points.T, size=size, face_color=face_color, edge_color=edge_color, **kwargs)
        return None
    
    def _add_labels(self, labels:Label, **kwargs):
        self.viewer.add_labels(labels, **kwargs)
        return None

viewer = napariWindow()
