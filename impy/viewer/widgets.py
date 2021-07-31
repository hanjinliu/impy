from __future__ import annotations
import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction, QPushButton, QWidget, QGridLayout
from .utils import *
from .._const import SetConst

# TODO: 
# - Integrate ImgArray functions after napari new version comes out. https://github.com/napari/napari/pull/263
# - Text layer -> https://github.com/napari/napari/issues/3053

__all__ = ["add_imread_menu",
           "add_imsave_menu",
           "add_table_widget", 
           "add_note_widget",
           "edit_properties",
           "function_handler",
           ]


FILTERS = ["None", "gaussian_filter", "median_filter", "mean_filter", "dog_filter", "log_filter", "erosion",
           "dilation", "opening", "closing", "entropy_filter", "std_filter", "coef_filter",
           "tophat", "rolling_ball"]

def add_imread_menu(viewer):
    from ..core import imread
    def open_img():
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window.qt_viewer,
            caption="Select file ...",
            directory=hist[0],
        )
        if (filenames != []) and (filenames is not None):
            img = imread(filenames[0])
            add_labeledarray(viewer, img)
        napari.utils.history.update_open_history(filenames[0])
        return None
    
    action = QAction('imread ...', viewer.window._qt_window)
    action.triggered.connect(open_img)
    viewer.window.file_menu.addAction(action)
    return None


def add_imsave_menu(viewer):
    def save_img():
        dlg = QFileDialog()
        layers = list(viewer.layers.selection)
        if len(layers) == 0:
            viewer.status = "Select a layer first."
            return None
        elif len(layers) > 1:
            viewer.status = "Select only one layer."
            return None
        img = layer_to_impy_object(viewer, layers[0])
        hist = napari.utils.history.get_save_history()
        dlg.setHistory(hist)
        
        if img.dirpath:
            last_hist = img.dirpath
        else:
            last_hist = hist[0]
        filename, _ = dlg.getSaveFileName(
            parent=viewer.window.qt_viewer,
            caption="Save image layer",
            directory=last_hist,
        )

        if filename:
            # TODO: multichannel?
            img.imsave(filename)
            napari.utils.history.update_save_history(filename)
        return None
    
    action = QAction('imsave ...', viewer.window._qt_window)
    action.triggered.connect(save_img)
    viewer.window.file_menu.addAction(action)
    return None


def edit_properties(viewer):
    """
    Edit properties of selected shapes or points.
    """    
    def edit_prop(event):
        # get the selected shape layer
        layers = list(viewer.layers.selection)
        if len(layers) != 1:
            return None
        layer = layers[0]
        if not isinstance(layer, (napari.layers.Points, napari.layers.Shapes)):
            return None
        
        old = layer.properties.get("text", [""]*len(layer.data))
        for i in layer.selected_data:
            try:
                old[i] = event.value.format(n=i).strip()
            except (ValueError, KeyError):
                old[i] = event.value.strip()
                
        layer.text.refresh_text({"text": old})
        return None
        
    line = magicgui.widgets.LineEdit(tooltip="Property editor")
    line.changed.connect(edit_prop)
    viewer.window.add_dock_widget(line, area="left", name="Property editor")
    return None


def add_table_widget(viewer):
    get_button = QPushButton("Get")
    @get_button.clicked.connect
    def make_table():
        dfs = list(iter_selected_layer(viewer, ["Points", "Tracks"]))
        if len(dfs) == 0:
            return
        for df in dfs:
            widget = QWidget()
            widget.setLayout(QGridLayout())
            axes = list(viewer.dims.axis_labels)
            columns = list(df.metadata.get("axes", axes[-df.data.shape[1]:]))
            table = magicgui.widgets.Table(df.data, name=df.name, columns=columns)
            copy_button = QPushButton("Copy")
            copy_button.clicked.connect(lambda: table.to_dataframe().to_clipboard())
            store_button = QPushButton("Store")
            @store_button.clicked.connect
            def _():
                viewer.window.results = table.to_dataframe()
                return None
            
            widget.layout().addWidget(table.native)
            widget.layout().addWidget(copy_button)
            widget.layout().addWidget(store_button)
            viewer.window.add_dock_widget(widget, area="right", name=df.name)
            
        return None
    
    viewer.window.add_dock_widget(get_button, area="left", name="Get Coordinates")
    return None
    

def add_note_widget(viewer):
    text = magicgui.widgets.TextEdit(tooltip="Note")
    text = viewer.window.add_dock_widget(text, area="right", name="Note")
    text.setVisible(False)
    return None

def add_filter(viewer):
    def add():
        @magicgui.magicgui(auto_call=True,
                           func={"choices": FILTERS, 
                                 "label": "function"},
                           param={"widget_type": "FloatSlider", 
                                  "min": 1, "max": 50, 
                                  "tooltip": "The first paramter, such as 'sigma' in gaussian_filter or 'radius' in median_filter"},
                           dims={"choices": ["2D", "3D"], 
                                 "tooltip": "Spatial dimension"},
                           fix_contrast_limits={"widget_type": "CheckBox", 
                                                "label": "fix contrast limits"},
                           layout="vertical")
        def _func(layer:napari.layers.Image, func, param=1, dims="2D", 
                  fix_contrast_limits=False) -> napari.types.LayerDataTuple:
            if layer is not None and func != "None":
                name = f"Result of {layer.name}"
                with SetConst("SHOW_PROGRESS", False):
                    try:
                        out = getattr(layer.data, func)(param, dims=int(dims[0]))
                    except Exception as e:
                        viewer.status = f"{func} finished with {e.__class__.__name__}: {e}"
                        return None
                try:
                    if fix_contrast_limits:
                        props_to_inherit = ["colormap", "blending", "translate", "scale", "contrast_limits"]
                    else:
                        props_to_inherit = ["colormap", "blending", "translate", "scale"]
                    kwargs = {k: getattr(viewer.layers[name], k, None) for k in props_to_inherit}
                except KeyError:
                    kwargs = dict(translate="inherit")
                    
                return _image_tuple(layer, out, name=name, **kwargs)
        
        viewer.window.add_dock_widget(_func, area="left", name="Filters")
        return None
    action = QAction("Filters", viewer.window._qt_window)
    action.triggered.connect(add)
    viewer.window.function_menu.addAction(action)
    return None


def add_threshold(viewer):
    def add():
        @magicgui.magicgui(auto_call=True,
                           percentile={"widget_type": "FloatSlider", "min": 0, "max": 100},
                           label={"widget_type": "CheckBox"},
                           layout="vertical")
        def _func(layer:napari.layers.Image, percentile=50, label=False) -> napari.types.LayerDataTuple:
            # define the name for the new layer
            if label:
                name = f"[L]{layer.name}"
            else:
                name = f"Threshold of {layer.name}"
                
            if layer is not None:
                with SetConst("SHOW_PROGRESS", False):
                    thr = np.percentile(layer.data, percentile)
                    if label:
                        out = layer.data.label_threshold(thr)
                        props_to_inherit = ["opacity", "blending", "translate", "scale"]
                        _as_layer_data_tuple = _label_tuple
                    else:
                        out = layer.data.threshold(thr)
                        props_to_inherit = ["colormap", "opacity", "blending", "translate", "scale"]
                        _as_layer_data_tuple = _image_tuple
                try:
                    kwargs = {k: getattr(viewer.layers[name], k, None) for k in props_to_inherit}
                except KeyError:
                    if label:
                        kwargs = dict(translate=layer.translate, opacity=0.3)
                    else:
                        kwargs = dict(translate=layer.translate, colormap="red", blending="additive")
                
                return _as_layer_data_tuple(layer, out, name=name, **kwargs)
        
        viewer.window.add_dock_widget(_func, area="left", name="Threshold/Label")
        return None
    action = QAction("Threshold/Label", viewer.window._qt_window)
    action.triggered.connect(add)
    viewer.window.function_menu.addAction(action)
    return None


def function_handler(viewer):
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
        layer_names = [l.name for l in viewer.layers]
        outlist = []
        i = 0
        for input in viewer.layers.selection:
            data = layer_to_impy_object(viewer, input)
            try:
                func = getattr(data, method)
            except AttributeError as e:
                viewer.status = f"{method} finished with AttributeError: {e}"
                continue
            
            viewer.status = f"{method} ..."
            try:
                args, kwargs = str_to_args(arguments)
                out = func(*args, **kwargs)
            except Exception as e:
                viewer.status = f"{method} finished with {e.__class__.__name__}: {e}"
                continue
            else:
                viewer.status = f"{method} finished"
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
                out_ = _image_tuple(input, out, name=name)
            elif isinstance(out, PhaseArray):
                out_ = (out, 
                        dict(scale=scale, name=name, colormap="hsv", translate=input.translate,
                                contrast_limits=out.border), 
                        "image")
            elif isinstance(out, Label):
                _label_tuple(input, out, name=name)
            elif isinstance(out, MarkerFrame):
                kw = dict(size=3.2, face_color=[0,0,0,0], translate=input.translate,
                          edge_color=viewer.window.cmap(),
                          metadata={"axes": str(out._axes), "scale": out.scale},
                          scale=scale)
                out_ = (out, kw, "points")
            elif isinstance(out, TrackFrame):
                out_ = (out, 
                        dict(scale=scale, translate=input.translate,
                             metadata={"axes": str(out._axes), "scale":out.scale}), 
                        "tracks")
            elif isinstance(out, PathFrame):
                out_ = (out, 
                        dict(scale=scale, translate=input.translate,
                             shape_type="path", edge_color="lime", edge_width=0.3,
                             metadata={"axes": str(out._axes), "scale":out.scale}),
                        "shapes")
            else:
                continue
            outlist.append(out_)
        
        if len(outlist) == 0:
            return None
        else:
            return outlist
    widget = viewer.window.add_dock_widget(run_func, area="left", name="Function Handler")
    widget.setVisible(False)
    return None


def _make_table_widget(df, columns=None, name=None):
    # DataFrame -> table
    widget = QWidget()
    widget.setLayout(QGridLayout())
    if columns is None:
        columns = list(df.columns)
    if name is None:
        name = "value"
    table = magicgui.widgets.Table(df.values, name=name, columns=columns)
    copy_button = QPushButton("Copy")
    copy_button.clicked.connect(lambda: table.to_dataframe().to_clipboard())                    
    widget.layout().addWidget(table.native)
    widget.layout().addWidget(copy_button)            
    return widget


def str_to_args(s:str) -> tuple[list, dict]:
    args_or_kwargs = list(_iter_args_and_kwargs(s))
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
    return eval(s, {"np": np})


def _iter_args_and_kwargs(string:str):
    stack = 0
    start = 0
    for i, s in enumerate(string):
        if s in ("(", "[", "{"):
            stack += 1
        elif s in (")", "]", "}"):
            stack -= 1
        elif stack == 0 and s == ",":
            yield string[start:i].strip()
            print(string[start:i].strip())
            start = i + 1
    if start == 0:
        yield string
        

def _image_tuple(input:napari.layers.Image, out:ImgArray, translate="inherit", **kwargs):
    data = input.data
    scale = make_world_scale(data)
    if out.dtype.kind == "c":
        out = np.abs(out)
    contrast_limits = [float(x) for x in out.range]
    if data.ndim == out.ndim:
        if isinstance(translate, str) and translate == "inherit":
            translate = input.translate
    elif data.ndim > out.ndim:
        if isinstance(translate, str) and translate == "inherit":
            translate = [input.translate[i] for i in range(data.ndim) if data.axes[i] in out.axes]
        scale = [scale[i] for i in range(data.ndim) if data.axes[i] in out.axes]
    else:
        if isinstance(translate, str) and translate == "inherit":
            translate = [0.0] + list(input.translate)
        scale = [1.0] + list(scale)
    kw = dict(scale=scale, colormap=input.colormap, translate=translate,
              blending=input.blending, contrast_limits=contrast_limits)
    kw.update(kwargs)
    return (out, kw, "image")

def _label_tuple(input:napari.layers.Labels, out:Label, translate="inherit", **kwargs):
    data = input.data
    scale = make_world_scale(data)
    if isinstance(translate, str) and translate == "inherit":
            translate = input.translate
    kw = dict(opacity=0.3, scale=scale, translate=translate)
    kw.update(kwargs)
    return (out, kw, "labels")