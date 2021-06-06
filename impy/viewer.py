from __future__ import annotations
import matplotlib.pyplot as plt
from .imgarray import ImgArray
from .labeledarray import LabeledArray
from .phasearray import PhaseArray
from .label import Label
from .specials import *
from .utilcls import ImportOnRequest
napari = ImportOnRequest("napari")
from magicgui import magicgui
from enum import Enum

# import napari.viewer
# TODO: 
# - Start different window if added object is apparently different. To do this, self.viewer should be
#   a property that returns the most recent window.
# - Layer does not remember the original data after c-split.
# - `bind_key` of cropping image
# - 

class Dims(Enum):
    ZYX = "zyx"
    YX = "yx"
    NONE = None
    
class napariWindow:
    _point_cmap = plt.get_cmap("rainbow", 16)
    _plot_cmap = plt.get_cmap("autumn", 16)
    
    def __init__(self):
        self._viewers = {}
        self._front_viewer = None
        self._point_color_id = 0
    
    def __repr__(self):
        w = "".join([f"<{k}>" for k in self._viewers.keys()])
        return f"{self.__class__}{w}"
    
    def __getitem__(self, key):
        """
        This method looks strange but intuitive because you can access the last viewer by
        >>> ip.window.add(...)
        while turn to another by
        >>> ip.window["X"].add(...)

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
    def selection(self):
        return [l.data for l in self.viewer.layers.selection]
    
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
                front = img # This is ImgArray
        if front is None:
            raise ValueError("There is no visible image layer.")
        return front
    
    @magicgui(call_button="Run", img={"label": "input:"})
    def run_func(self,
                 img: napari.layers.Image,
                 method="gaussian_filter", 
                 firstparam="None",
                 dims=Dims.NONE,
                 update=False) -> napari.types.LayerDataTuple:
        """
        Run image analysis in napari window.

        Parameters
        ----------
        img : napari.layers.Image
            Input image layer.
        method : str, default is "gaussian_filter"
            Name of method to be called.
        firstparam : str, optional
            First parameter if exists.
        dims : str, optional
            Spatial dimensions.

        Returns
        -------
        napari.types.LayerDataTuple
            This is passed to napari and is directly visualized.
        """        
        if img is None:
            return None
        
        if firstparam == "None":
            try:
                out = getattr(img.data, method)(dims=dims.value)
            except TypeError:
                out = getattr(img.data, method)()
        else:
            try:
                firstparam = float(firstparam)
            except ValueError:
                pass
            out = getattr(img.data, method)(float(firstparam), dims=dims.value)
        scale = make_world_scale(img.data)
        
        # determine name of the new layer
        if update and type(img.data) is type(out):
            name = img.name
        else:
            layer_names = [l.name for l in self.layers]
            name = method
            i = 0
            while name in layer_names:
                name = f"{method}-{i}"
                
        if isinstance(out, ImgArray):
            return (out, 
                    dict(scale=scale, name=name, colormap=img.colormap, 
                         contrast_limits=[float(x) for x in out.range]), 
                    "image")
        elif isinstance(out, PhaseArray):
            return (out, 
                    dict(scale=scale, name=name, colormap="hsv"), 
                    "image")
        elif isinstance(out, Label):
            return (out, 
                    dict(opacity=0.3, scale=scale, name=name), 
                    "labels")
        elif isinstance(out, MarkerFrame):
            cmap = self.__class__._point_cmap
            kw = dict(size=3.2, face_color=[0,0,0,0], 
                      edge_color=list(cmap(self._point_color_id * (cmap.N//2+1) % cmap.N)), 
                      scale=scale)
            self._point_color_id += 1
            return (out, kw, "points")
        
    def start(self, key:str):
        """
        Create a napari window with name `key`.
        """        
        if not isinstance(key, str):
            raise TypeError("`key` must be str.")
        if key in self._viewers.keys():
            raise ValueError(f"Key {key} already exists.")
        viewer = napari.Viewer(title=key)
        default_viewer_settings(viewer)
        viewer.window.add_dock_widget(self.run_func, area="bottom")
        cropfunc = lambda viewer: self.crop(dims="tzc", viewer=viewer)
        viewer.bind_key("Control-Shift-X")(cropfunc)
        self._viewers[key] = viewer
        self._front_viewer = key
        return None
    
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
        # TODO: does not raise Error by default, but:
        # - for multi-channel, label is discarded because destination is a slice of original image
        # - when destination's shape is wrong, error.
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
        
    def crop(self, dims:str="tzc", viewer:napari.Viewer=None):
        """
        Crop images at the edges of the napari viewer. This function can be called with key binding by
        default.

        Parameters
        ----------
        dims : str, optional
            Which axes will not be cropped. Generally when an image stack is cropped at the edges, all 
            the images along t-axis should be cropped at the same edges. On the other hand, images at
            different positions (different p-coordinates) should not. That is why default is "tzc".
        viewer : napari.Viewer, optional
            Target viewer.
        """        
        if viewer is None:
            viewer = self.viewer
            
        imglist = list(filter(lambda x: isinstance(x, napari.layers.Image), viewer.layers.selection))
        count = 0
        for layer in imglist:
            sl = []
            translate = []
            for i, (start, end) in enumerate(layer.corner_pixels.T):
                start, end = int(start), int(end+1)
                if start+1 < end:
                    if layer.data.axes[i] in "yx":
                        translate.append(start*layer.data.scale[layer.data.axes[i]])
                    sl.append(slice(start, end))
                else:
                    if layer.data.axes[i] in dims:
                        sl.append(slice(None))
                    else:
                        sl.append(start)
            
            img = layer.data[tuple(sl)]
            if img.size > 0:
                self._add_image(img, translate=translate, blending=layer.blending, colormap=layer.colormap)
                count += 1
        
        if count == 0:
            raise ValueError("No image was cropped")
        
        
    
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
            layer_type = napari.layers.Shapes
        elif layer_type == "image":
            layer_type = napari.layers.Image
        elif layer_type == "point":
            layer_type = napari.layers.Points
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
            names = [f"[L]{name}"] * len(lbls)
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
            for sl, data in df.groupby(groupax): # TODO: doesn't work for []
                path = data.values.tolist()
                paths.append(path)
                ec.append(list(cmap(self._point_color_id * (cmap.N//2+1) % cmap.N)))
                self._point_color_id += 1
            
            kw = dict(edge_width=0.8, opacity=0.75, scale=scale, edge_color=ec, face_color=ec)
            kw.update(kwargs)
            self._viewers["Plot"].add_shapes(paths, shape_type="path", **kw)
        
        new_axes = list(df.columns)
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self._viewers["Plot"].dims.axis_labels):
            self._viewers["Plot"].dims.axis_labels = new_axes
        
        return None

    def _name(self, name="impy"):
        i = 0
        existing = self._viewers.keys()
        while name in existing:
            name += f"-{i}"
            i += 1
        return name
        
        

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

def default_viewer_settings(viewer):
    viewer.scale_bar.visible = True
    viewer.scale_bar.ticks = False
    viewer.scale_bar.font_size = 8
    viewer.axes.visible = True
    viewer.axes.colored = False
    return None

window = napariWindow()
