from __future__ import annotations
import matplotlib.pyplot as plt
from .bases import MetaArray
from .labeledarray import LabeledArray
from .phasearray import PhaseArray
from .label import Label
from .specials import *
from .utilcls import ImportOnRequest
napari = ImportOnRequest("napari")

# TODO: 
# - different name (different scale or shape) for different window?
# crop
# crop = ip.window.last_layer.data[tuple(slice(int(i[0]), int(i[1])) for i in ip.window.last_layer.corner_pixels.T)]

class napariWindow:
    _point_cmap = plt.get_cmap("rainbow", 16)
    _plot_cmap = plt.get_cmap("autumn", 16)
    
    def __init__(self):
        self.viewer = None
        self._point_color_id = 0
        
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
    
    @property
    def front_image(self):
        """
        From list of image layers return the most front visible image.
        """        
        front = None
        for img in self.iter_layer("image"):
            if img.visible:
                front = img # This is a view of ImgArray
        if front is None:
            raise ValueError("There is no visible image layer.")
        return front
        
    def start(self):
        self.viewer = napari.Viewer(title="impy")
        self.viewer.scale_bar.visible = True
        self.viewer.scale_bar.ticks = False
        self.viewer.scale_bar.font_size = 8
        self.viewer.axes.visible = True
        self.viewer.axes.colored = False
    
    def add(self, obj, **kwargs):
        """
        Add images, points, labels, tracks or graph to viewer.

        Parameters
        ----------
        obj : Any
            Object to add.
        """        
        self.viewer is None and self.start()
        if isinstance(obj, LabeledArray):
            self._add_image(obj, **kwargs)
        elif isinstance(obj, MarkerFrame):
            self._add_points(obj, **kwargs)
        elif isinstance(obj, Label):
            self._add_labels(obj, **kwargs)
        elif isinstance(obj, TrackFrame):
            self._add_tracks(obj, **kwargs)
        elif isinstance(obj, PropArray):
            self._add_plot(obj, **kwargs)
        else:
            raise TypeError(f"Could not interpret type: {type(obj)}")
                
    def shapes_to_labels(self, destination:LabeledArray=None, index:int=0, projection:bool=False):
        """
        Convert manually drawn shapes to labels and store in `destination`.

        Parameters
        ----------
        destination : LabeledArray, optional
            To which labels will be stored, by default None
        index : int, default is 0
            Index of shape layer. This needs consideration only if there are multiple shape layers.
        projection : bool, default is False
            If collect all the shapes in different layers by projection. This argument is taken into
            account only if there are multiple shape layers.

        Returns
        -------
        Label
            Label image that was made from shapes.
        """        
        # TODO: does not work for multichannel images because axes is aborted by np.take
        if destination is None:
            destination = self.front_image.data
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
    
    def points_to_frames(self, ref:LabeledArray=None, index=0, projection=False) -> MarkerFrame:
        """
        Convert manually selected points to MarkerFrame.

        Parameters
        ----------
        ref : LabeledArray, optional
            Reference image to determine extent of point coordinates.
        index : int, optional
            Index of point layer. This needs consideration only if there are multiple point layers.
        projection : bool, default is False
            If collect all the points in different layers by projection. This argument is taken into
            account only if there are multiple point layers.

        Returns
        -------
        MarkerFrame
            DataFrame of points.
        """        
        if ref is None:
            ref = self.front_image.data
        zoom_factors = [self.scale[a]/ref.scale[a] for a in ref.axes]
        points = [points.data/zoom_factors for points in self.iter_layer("point")]
        if not projection:
            data = points[index]
        else:
            data = np.vstack(points)
        mf = MarkerFrame(data, columns = self.axes)
        mf.set_scale(self)
        return mf
    
    def crop_front_image(self, dims="tzc"):
        layer = self.front_image
        sl = []
        for i, (start, end) in enumerate(layer.corner_pixels.T):
            start, end = int(start), int(end)
            if start+1 < end:
                sl.append(slice(start, end))
            else:
                if layer.data.axes[i] in dims:
                    sl.append(slice(None))
                else:
                    sl.append(start)
        return layer.data[tuple(sl)]
    
    def iter_layer(self, layer_type:str):
        """
        Iterate over layers and yield only certain type of layers.

        Parameters
        ----------
        layer_type : str, {"shape", "image", "point"}
            Type of layer.

        Yields
        -------
        napari.layers
            Layers specified by layer_type
        """        
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
        
    def _add_image(self, img:LabeledArray, **kwargs):
        chn_ax = img.axisof("c") if "c" in img.axes else None
        
        if isinstance(img, PhaseArray) and not "colormap" in kwargs.keys():
            kwargs["colormap"] = "hsv"
            kwargs["contrast_limits"] = img.border
        
        scale = make_world_scale(img)
        
        if len(img.history) > 0:
            suffix = "-" + img.history[-1]
        else:
            suffix = ""
        
        name = "No-Name" if img.name is None else img.name
        if chn_ax is not None:
            name = [f"[C{i}]{name}{suffix}" for i in range(img.sizeof("c"))]
        else:
            name = [name + suffix]
        
        self.viewer.add_image(img, channel_axis=chn_ax, scale=scale, 
                              name=name if len(name)>1 else name[0],
                              **kwargs)
        self.viewer.scale_bar.unit = img.scale_unit
        if hasattr(img, "labels"):
            self._add_labels(img.labels, name=name)
        
        new_axes = [a for a in img.axes if a != "c"]
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self.viewer.dims.axis_labels):
            self.viewer.dims.axis_labels = new_axes
        return None
    
    def _add_points(self, points, **kwargs):
        if isinstance(points, MarkerFrame):
            # scale = [points.scale[a] for a in points._axes if a != "c"]
            scale = make_world_scale(points)
            points = points.get_coords()
        else:
            scale=None
        
        cmap = self.__class__._point_cmap
        if "c" in points._axes:
            pnts = points.split("c")
        else:
            pnts = [points]
            
        for each in pnts:
            kw = dict(size=3.2, face_color=[0,0,0,0], 
                      edge_color=list(cmap(self._point_color_id * (cmap.N//2+1) % cmap.N)))
            kw.update(kwargs)
            self.viewer.add_points(each.values, scale=scale, **kw)
            self._point_color_id += 1
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
            names = [f"[L]{name}"]*len(lbls)
        else:
            names = [labels.name]
            
        for lbl, name in zip(lbls, names):
            self.viewer.add_labels(lbl, opacity=opacity, scale=scale, name=name, **kwargs)
        return None

    def _add_tracks(self, track:TrackFrame, **kwargs):
        if "c" in track.col_axes:
            track_list = track.split("c")
        else:
            track_list = [track]
            
        scale = make_world_scale(track[[a for a in track._axes if a != "p"]])
        for tr in track_list:
            self.viewer.add_tracks(tr, scale=scale, **kwargs)
        
        return None

    def _add_plot(self, prop:PropArray, **kwargs):
        input_df = prop.as_frame()
        if "c" in input_df.columns:
            dfs = input_df.split("c")
        else:
            dfs = [input_df]
        
        if len(dfs[0].columns) > 2:
            groupax = find_first_appeared(input_df.col_axes, include="ptz<yx")
        else:
            groupax = []
        
        for df in dfs:
            maxima = df.max(axis=0).values
            order = list(np.argsort(maxima))
            df = df[df.columns[order]]
            maxima = maxima[order]
            scale = [1] * maxima.size
            scale[-1] = max(maxima[:-1])/maxima[-1]
            cmap = self.__class__._plot_cmap
            paths = []
            ec = []
            for sl, data in df.groupby(groupax):
                path = data.values.tolist()
                paths.append(path)
                ec.append(list(cmap(self._point_color_id * (cmap.N//2+1) % cmap.N)))
                self._point_color_id += 1
            
            kw = dict(edge_width=0.8, opacity=0.75, scale=scale, edge_color=ec, face_color=ec)
            kw.update(kwargs)
            self.viewer.add_shapes(paths, shape_type="path", **kw)
        
        new_axes = list(df.columns)
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self.viewer.dims.axis_labels):
            self.viewer.dims.axis_labels = new_axes
        
        return None

def to_labels(layer, labels_shape, zoom_factor=1):
    return layer._data_view.to_labels(labels_shape=labels_shape, zoom_factor=zoom_factor)
    

def make_world_scale(obj):
    scale = []
    for a in obj._axes:
        if a in "zyx":
            scale.append(obj.scale[a])
        elif a == "c":
            pass
        else:
            scale.append(1)
    return scale

window = napariWindow()
