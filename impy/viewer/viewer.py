from __future__ import annotations
import matplotlib.pyplot as plt
from ..imgarray import ImgArray
from ..labeledarray import LabeledArray
from ..phasearray import PhaseArray
from ..label import Label
from ..specials import *
from ..utilcls import ImportOnRequest
from .utils import *
from .mouse import *
# from matplotlib.backends.backend_qt5agg import FigureCanvas
# from matplotlib.figure import Figure

magicgui = ImportOnRequest("magicgui")

# import napari.viewer
# TODO: 
# - Layer does not remember the original data after c-split.
# - plot widget

        
class napariWindow:
    _point_cmap = plt.get_cmap("rainbow", 16)
    
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
    def selection(self) -> list:
        return list(map(get_data, self.viewer.layers.selection))
    
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
        if not self._viewers:
            from . import keybinds
        
        viewer = napari.Viewer(title=key)
        default_viewer_settings(viewer)
        # Add dock widgets
        viewer.window.add_dock_widget(self._function_handler(), area="left", name="Function Handler")
        viewer.window.add_dock_widget(magicgui.widgets.TextEdit(), area="right", name="memo")
        viewer.window.add_dock_widget(self._table(), area="right", name="table")
        # Add event
        viewer.layers.events.inserted.connect(upon_add_layer)
        # Add menu
        viewer.window.file_menu.addSeparator()
        self._add_imread_menu(viewer)
        
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
        points = [points.data/zoom_factors for points in self._iter_layer("Points")]
        if not projection:
            data = points[index]
        else:
            data = np.vstack(points)
        mf = MarkerFrame(data, columns = self.axes)
        mf.set_scale(self)
        return mf
    
        
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    #    Others
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
    
    def _iter_layer(self, layer_type:str):
        return iter_layer(self.viewer, layer_type)
    
    def _iter_selected_layer(self, layer_type:str):
        return iter_selected_layer(self.viewer, layer_type)
        
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
            metadata = {"axes": str(each._axes), "scale": each.scale}
            kw = dict(size=3.2, face_color=[0,0,0,0], metadata=metadata,
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
        if "c" in track._axes:
            track_list = track.split("c")
        else:
            track_list = [track]
            
        scale = make_world_scale(track[[a for a in track._axes if a != "p"]])
        for tr in track_list:
            metadata = {"axes": str(tr._axes), "scale": tr.scale}
            self.viewer.add_tracks(tr, scale=scale, metadata=metadata, **kwargs)
        
        return None

    def __add_plot(self, prop:PropArray, **kwargs):
        # TODO: Delete this. Use magicgui for plotting inside
        input_df = prop.as_frame()
        if "c" in input_df.columns:
            dfs = input_df.split("c")
        else:
            dfs = [input_df]
        
        if len(dfs[0].columns) > 2:
            groupax = find_first_appeared(input_df._axes, include="ptz<yx")
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
            self["Plot"].viewer.add_shapes(paths, shape_type="path", **kw)
        
        new_axes = list(df.columns)
        # add axis labels to slide bars and image orientation.
        if len(new_axes) >= len(self["Plot"].viewer.dims.axis_labels):
            self["Plot"].viewer.dims.axis_labels = new_axes
        
        return None
    
    # def _add_plot(self, data):
        # static_canvas = FigureCanvas(Figure(figsize=(5, 3)))
        # axes = static_canvas.figure.subplots()
        # axes.plot(data)
        # TODO: write this

    def _name(self, name="impy"):
        i = 0
        existing = self._viewers.keys()
        while name in existing:
            name += f"-{i}"
            i += 1
        return name
    
    def _table(self):
        # TODO: 
        # - this should be keybinding of points/tracks layers 
        # - add in a tab
        @magicgui.magicgui(call_button="Get")
        def make_table():
            dfs = list(self._iter_selected_layer(["Points", "Tracks"]))
            if len(dfs) == 0:
                return
            for df in dfs:
                columns = list(df.metadata["axes"])
                table = magicgui.widgets.Table(df.data, name=df.name, columns=columns)
                self.viewer.window.add_dock_widget(table, area="right", name=df.name)
            return None
        
        return make_table
    
    def _add_imread_menu(self, viewer):
        from qtpy.QtWidgets import QFileDialog, QAction
        from ..core import imread
        def open_img():
            dlg = QFileDialog()
            hist = napari.utils.history.get_open_history()
            dlg.setHistory(hist)
            filenames, _ = dlg.getOpenFileNames(
                parent=self.viewer.window.qt_viewer,
                caption='Select file ...',
                directory=hist[0],
            )
            if (filenames != []) and (filenames is not None):
                img = imread(filenames[0])
                self.add(img)
            napari.utils.history.update_open_history(filenames[0])
            return None
        action = QAction('imread ...', viewer.window._qt_window)
        action.triggered.connect(open_img)
        viewer.window.file_menu.addAction(action)
        return None
    
    def _function_handler(self):
        @magicgui.magicgui(call_button="Run")
        def run_func(method="gaussian_filter", 
                     arguments="",
                     update=False) -> napari.types.LayerDataTuple:
            """
            Run image analysis in napari window.

            Parameters
            ----------
            method : str, default is "gaussian_filter"
                Name of method to be called.
            arguments : str, default is ""
                Input arguments and keyword arguments. If you want to run `self.median_filter(2, dims=2)` then
                the value should be `"2, dims=2"`.
            update : bool, default is False
                If update the layer's data. The original data will NOT be updated.
                
            Returns
            -------
            napari.types.LayerDataTuple
                This is passed to napari and is directly visualized.
            """
            try:
                inputs = list(l for l in self.viewer.layers.selection)
            except StopIteration: # ???
                return None
            layer_names = [l.name for l in self.viewer.layers]
            outlist = []
            i = 0
            for input in inputs:
                data = get_data(input)
                try:
                    func = getattr(data, method)
                except AttributeError as e:
                    self.viewer.status = f"{method} finished with AttributeError: {e}"
                    continue
                
                self.viewer.status = f"{method} ..."
                try:
                    args, kwargs = str_to_args(arguments)
                    out = func(*args, **kwargs)
                except Exception as e:
                    self.viewer.status = f"{method} finished with {e.__class__.__name__}: {e}"
                    continue
                else:
                    self.viewer.status = f"{method} finished"
                scale = make_world_scale(data)
                
                # determine name of the new layer
                if update and type(data) is type(out):
                    name = input.name
                else:
                    name = f"{method}-{i}"
                    i += 1
                    while name in layer_names:
                        name = f"{method}-{i}"
                        i += 1
                
                if isinstance(out, ImgArray):
                    contrast_limits = [float(x) for x in out.range]
                    out_ = (out, 
                            dict(scale=scale, name=name, colormap=input.colormap, translate=input.translate,
                                 blending=input.blending, contrast_limits=contrast_limits), 
                            "image")
                elif isinstance(out, PhaseArray):
                    out_ = (out, 
                            dict(scale=scale, name=name, colormap="hsv", contrast_limits=out.border), 
                            "image")
                elif isinstance(out, Label):
                    out_ = (out, 
                            dict(opacity=0.3, scale=scale, name=name), 
                            "labels")
                elif isinstance(out, MarkerFrame):
                    cmap = self.__class__._point_cmap
                    kw = dict(size=3.2, face_color=[0,0,0,0], 
                                edge_color=list(cmap(self._point_color_id * (cmap.N//2+1) % cmap.N)),
                                metadata={"axes": str(out._axes), "scale": out.scale},
                                scale=scale)
                    self._point_color_id += 1
                    out_ = (out, kw, "points")
                elif isinstance(out, TrackFrame):
                    out_ = (out, 
                            dict(scale=scale, metadata={"axes": str(out._axes), "scale":out.scale}), 
                            "tracks")
                else:
                    continue
                outlist.append(out_)
            
            if len(outlist) == 0:
                return None
            else:
                return outlist
            
        return run_func  
    
def default_viewer_settings(viewer):
    viewer.scale_bar.visible = True
    viewer.scale_bar.ticks = False
    viewer.scale_bar.font_size = 8
    viewer.axes.visible = True
    viewer.axes.colored = False
    return None

def str_to_args(s:str) -> tuple[list, dict]:
    args_or_kwargs = [s.strip() for s in s.split(",")]
    if args_or_kwargs[0] == "":
        return [], {}
    args = []
    kwargs = {}
    for a in args_or_kwargs:
        if "=" in a:
            k, v = a.split("=")
            v = interpret_type(v)
            kwargs[k] = v
        else:
            a = interpret_type(a)
            args.append(a)
    return args, kwargs
            
def interpret_type(s:str):
    try:
        s = int(s)
    except ValueError:
        try:
            s = float(s)
        except ValueError:
            pass
    return s

