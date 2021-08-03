from __future__ import annotations
import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction, QPushButton, QWidget, QGridLayout
from .widgets import *
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
           "add_rectangle_editor",
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
    controller_widget = Controller(viewer)
    viewer.window.add_dock_widget(controller_widget, area="left", name="impy controller")
    return None
    
def add_note_widget(viewer):
    text = magicgui.widgets.TextEdit(tooltip="Note")
    text = viewer.window.add_dock_widget(text, area="right", name="Note")
    text.setVisible(False)
    return None

def add_gui_to_function_menu(viewer, gui, name):
    action = QAction(name, viewer.window._qt_window)
    @action.triggered.connect
    def _():
        if name in viewer.window._dock_widgets:
            viewer.window._dock_widgets[name].show()
        else:
            viewer.window.add_dock_widget(gui(viewer), area="left", name=name)
        return None
    
    viewer.window.function_menu.addAction(action)
    return None

def add_filter(viewer):
    return add_gui_to_function_menu(viewer, FunctionCaller, "Filters")

def add_threshold(viewer):
    return add_gui_to_function_menu(viewer, ThresholdAndLabel, "Threshold/Label")

def add_rectangle_editor(viewer):
    return add_gui_to_function_menu(viewer, RectangleEditor, "Rectangle Editor")

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
                    out_ = image_tuple(input, out, name=name)
                elif isinstance(out, PhaseArray):
                    out_ = (out, 
                            dict(scale=scale, name=name, colormap="hsv", translate=input.translate,
                                    contrast_limits=out.border), 
                            "image")
                elif isinstance(out, Label):
                    label_tuple(input, out, name=name)
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
        