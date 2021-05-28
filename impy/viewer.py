import matplotlib.pyplot as plt
from .metaarray import MetaArray
from .labeledarray import LabeledArray
from .phasearray import PhaseArray
from .label import Label
from .specials import *
from .utilcls import ImportOnRequest
napari = ImportOnRequest("napari")

"""
# marker
viewer.layers[-1].name="peaks"
a=viewer.layers[-1]
a.symbol="arrow"
a.size = 0.2
"""

# TODO: 
# - different name (different scale or shape) for different window?
# - show PropArray as 3D plot (need new window anyway)

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
    
    @property
    def axes(self):
        return "".join(self.viewer.dims.axis_labels)
    
    @property
    def scale(self):
        d = self.viewer.dims
        return {a: r[2] for a, r in zip(d.axis_labels, d.range)}
        
    def start(self):
        self.viewer = napari.Viewer(title="impy")
        self.viewer.scale_bar.visible=True
        self.viewer.axes.visible=True
    
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
                
    def shapes_to_labels(self, destination:LabeledArray=None, index=0, projection=False):
        if destination is None:
            destination = self.get_front_image()
        zoom_factors = [self.scale[a]/destination.scale[a] for a in "yx"]
        if np.unique(zoom_factors).size == 1:
            zoom_factor = zoom_factors[0]
        else:
            raise ValueError("Scale mismatch in images and napari world.")
        
        shapes = [to_labels(layer, destination.shape, zoom_factor=zoom_factor) 
                  for layer in self.iter_layer("shape")]
        
        if not projection:
            label = shapes[index]
        else:
            label = np.sum(shapes, axis=0)
        if hasattr(destination, "labels"):
            print("Label already exist. Overlapped.")
            del destination.labels
        destination.append_label(label)
        return destination.labels
    
    def points_to_frames(self, ref:LabeledArray=None, index=0, projection=False):
        if ref is None:
            ref = self.get_front_image()
        zoom_factors = [self.scale[a]/ref.scale[a] for a in ref.axes]
        points = [points.data/zoom_factors for points in self.iter_layer("point")]
        if not projection:
            data = points[index]
        else:
            data = np.vstack(points)
        mf = MarkerFrame(data, columns = self.axes)
        mf.set_scale(self)
        return mf
    
    def iter_layer(self, layer_type:str):
        if layer_type == "shape":
            layer_type = napari.layers.shapes.shapes.Shapes
        elif layer_type == "image":
            layer_type = napari.layers.image.Image
        elif layer_type == "point":
            layer_type = napari.layers.points.Points
        else:
            raise NotImplementedError
        
        for layer in self.layers:
            if isinstance(layer, layer_type):
                yield layer
    
    def get_front_image(self):
        front = None
        for img in self.iter_layer("image"):
            if img.visible:
                front = img.data
        if front is None:
            raise ValueError("There is no visible image layer.")
        return front
        
    def _add_image(self, img:LabeledArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
        
        if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
            kwargs["colormap"] = "hsv"
            kwargs["contrast_limits"] = img.border
        
        scale = make_world_scale(img)
                
        self.viewer.add_image(img, channel_axis=chn_ax, scale=scale, name=img.name,
                              **kwargs)
        
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=f"Label of {img.name}")
        
        new_axes = [a for a in img.axes if a != "c"]
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self.viewer.dims.axis_labels):
            self.viewer.dims.axis_labels = new_axes
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
    
    def _add_labels(self, labels:Label, opacity:float=0.3, **kwargs):
        scale = make_world_scale(labels)
        if "c" in labels.axes:
            lbls = labels.split("c")
        else:
            lbls = [labels]
            
        for lbl in lbls:
            self.viewer.add_labels(lbl, opacity=opacity, scale=scale, **kwargs)
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

            
def get_axes(obj):
    if isinstance(obj, MetaArray):
        return obj.axes
    elif isinstance(obj, AxesFrame):
        return obj.col_axes
    else:
        return None

def to_labels(layer, labels_shape, zoom_factor=1):
    return layer._data_view.to_labels(labels_shape=labels_shape, zoom_factor=zoom_factor)
    

def make_world_scale(img):
    scale = []
    for a in img.axes:
        if a in "zyx":
            scale.append(img.scale[a])
        elif a == "c":
            pass
        else:
            scale.append(1)
    return scale

window = napariWindow()
