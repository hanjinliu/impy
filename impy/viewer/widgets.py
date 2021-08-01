from __future__ import annotations
import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction, QPushButton, QWidget, QGridLayout, QHBoxLayout
from .utils import *
from .._const import SetConst

__all__ = ["add_imread_menu",
           "add_imsave_menu",
           "add_controller_widget", 
           "add_note_widget",
           "edit_properties",
           "add_threshold", 
           "add_filter", 
           "add_regionprops",
           "function_handler",
           ]


FILTERS = ["None", "gaussian_filter", "median_filter", "mean_filter", "dog_filter", "doh_filter", "log_filter", 
           "erosion", "dilation", "opening", "closing", "entropy_filter", "std_filter", "coef_filter",
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


def add_controller_widget(viewer):
    # Convert Points/Tracks layer into a table widget
    get_button = QPushButton("(x,y)")
    @get_button.clicked.connect
    def _():
        """
        widget = table widget + button widget
        ---------------
        |             |
        |   (table)   | <- table widget
        |             |
        |[Copy][Store]| <- button widget = copy button + store button
        ---------------
        """        
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
            
            button_widget = QWidget()
            layout = QHBoxLayout()
            layout.addWidget(copy_button)
            layout.addWidget(store_button)
            button_widget.setLayout(layout)
            
            widget.layout().addWidget(table.native)
            widget.layout().addWidget(button_widget)
            viewer.window.add_dock_widget(widget, area="right", name=df.name)
            
        return None
    
    # Add text layer
    add_button = QPushButton("+Text")
    @add_button.clicked.connect
    def _():
        layer = viewer.add_shapes(ndim=2, shape_type="rectangle", name="Text Layer")
        layer.mode = "add_rectangle"
        layer.current_edge_width = 2.0 # unit is pixel here
        layer.current_face_color = [0, 0, 0, 0]
        layer.current_edge_color = [0, 0, 0, 0]
        layer._rotation_handle_length = 20/np.mean(layer.scale[-2:])
        layer.current_properties = {"text": np.array(["text here"], dtype="<U32")}
        layer.properties = {"text": np.array([], dtype="<U32")}
        layer.text = "{text}"
        layer.text.size = 6.0 * Const["FONT_SIZE_FACTOR"]
        layer.text.color = "white"
        layer.text.anchor = "center"
        return None
    
    # Add Labels layer that is connected to ImgArray
    label_button = QPushButton("Label")
    @label_button.clicked.connect
    def _():
        selected = list(viewer.layers.selection)
        if len(selected) != 1:
            return None
        selected = selected[0]
        if not isinstance(selected, napari.layers.Image):
            return None
        img = selected.data
        if hasattr(img, "labels"):
            return None
        with SetConst("SHOW_PROGRESS", False):
            img.append_label(np.zeros(img.shape, dtype=np.uint8))
        layer = viewer.add_labels(img.labels.value, opacity=0.3, scale=selected.scale, name=f"[L]{img.name}",
                                  translate=selected.translate)
        layer.mode = "paint"
        return None
        
    controller_widget = QWidget()
    layout = QHBoxLayout()
    layout.addWidget(get_button)
    layout.addWidget(add_button)
    layout.addWidget(label_button)
    controller_widget.setLayout(layout)
    viewer.window.add_dock_widget(controller_widget, area="left", name="impy controller")
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
                                  "min": 1, "max": 30, 
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
                           percentile={"widget_type": "FloatSlider", 
                                       "min": 0, "max": 100,
                                       "tooltip": "Threshold percentile"},
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

def add_regionprops(viewer):
    # TODO: close main window after calculation, choose properties
    @magicgui.magicgui(main_window=True,
                       call_button="Calculate")
    def regionprops():
        selected = list(viewer.layers.selection)
        if len(selected) < 1:
            return None
        properties = ("label", "mean_intensity")
        for imglayer in selected:
            if not isinstance(imglayer, napari.layers.Image):
                continue
            lbl = imglayer.data.labels
            with SetConst("SHOW_PROGRESS", False):
                out = imglayer.data.regionprops(properties=properties)
            out["label"] = out["label"].astype(lbl.dtype)
            order = np.argsort(out["label"].value)
            d = {k: np.concatenate([[0], out[k].value[order]]) for k in properties}
            # find Labels layer
            for l in viewer.layers:
                if l.data is lbl:
                    l.properties = d
                    break
            else:
                l = viewer.add_labels(lbl.value, opacity=0.3, scale=imglayer.scale, 
                                      name=f"[L]{imglayer.name}", translate=imglayer.translate)
    
    action = QAction("regionprops", viewer.window._qt_window)
    action.triggered.connect(lambda: regionprops.show(run=True))
    viewer.window.function_menu.addAction(action)
    return None

def function_handler(viewer):
    def add():
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
            outlist = []
            for input in viewer.layers.selection:
                data = layer_to_impy_object(viewer, input)
                try:
                    if method.startswith("[") and method.endswith("]"):
                        arguments = method[1:-1]
                        func = data.__getitem__
                    else:
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
                    name = f"Result of {input.name}"
                        
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
        viewer.window.add_dock_widget(run_func, area="left", name="Function Handler")
        return None
    action = QAction("Function Handler", viewer.window._qt_window)
    action.triggered.connect(add)
    viewer.window.function_menu.addAction(action)
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
        if "=" in a and a[0] not in ("'", '"'):
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