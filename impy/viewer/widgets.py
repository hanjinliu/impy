from __future__ import annotations
import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction, QPushButton, QWidget, QGridLayout
from .utils import *

# TODO: 
# - Integrate ImgArray functions after napari new version comes out. https://github.com/napari/napari/pull/263
# - Text layer -> https://github.com/napari/napari/issues/3053

__all__ = ["add_imread_menu",
           "add_imsave_menu",
           "add_table_widget", 
           "add_note_widget",
           "function_handler"]

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
    @magicgui.magicgui(call_button="Apply")
    def edit_prop(format_="{text}", propname="text", value=""):
        # get the selected shape layer
        layers = list(viewer.layers.selection)
        if len(layers) != 1:
            return None
        layer = layers[0]
        if not isinstance(layer, (napari.layers.Labels, napari.layers.Points, napari.layers.Shapes)):
            return None
        
        # current properties
        # props = layer.properties
        # if prop not in layer.properties.keys():
        props = np.zeros(len(layer.data), dtype="<U12")
        old = layer.properties.get(propname, [""]*len(props))
        # new format of texts
        layer.text._text_format_string = format_
        # set new values to selected shapes
        for i in range(len(props)):
            if layer.selected_data:
                props[i] = value
            else:
                props[i] = old[i]
                
        layer.properties = {propname: props}
        layer.text.refresh_text(props)
    
    viewer.window.add_dock_widget(edit_prop, area="right", name="Property editor")
    return None

def add_table_widget(viewer):
    QtViewerDockWidget = napari._qt.widgets.qt_viewer_dock_widget.QtViewerDockWidget
    
    button = QPushButton("Get")
    @button.clicked.connect
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
            widget.layout().addWidget(table.native)
            widget.layout().addWidget(copy_button)
            widget = QtViewerDockWidget(viewer.window.qt_viewer, table, name=df.name,
                                        area="right", add_vertical_stretch=True)
            viewer.window._add_viewer_dock_widget(widget, tabify=viewer.window.n_table>0)
            viewer.window.n_table += 1
        return None
    
    viewer.window.add_dock_widget(button, area="left", name="Get Coordinates")
    return None
    

def add_note_widget(viewer):
    text = magicgui.widgets.TextEdit(tooltip="Note")
    text = viewer.window.add_dock_widget(text, area="right", name="Note")
    text.setVisible(False)
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
                if out.dtype.kind == "c":
                    out = np.abs(out)
                contrast_limits = [float(x) for x in out.range]
                if data.ndim == out.ndim:
                    translate = input.translate
                elif data.ndim > out.ndim:
                    translate = [input.translate[i] for i in range(data.ndim) if data.axes[i] in out.axes]
                    scale = [scale[i] for i in range(data.ndim) if data.axes[i] in out.axes]
                else:
                    translate = [0.0] + list(input.translate)
                    scale = [1.0] + list(scale)
                out_ = (out, 
                        dict(scale=scale, name=name, colormap=input.colormap, translate=translate,
                             blending=input.blending, contrast_limits=contrast_limits), 
                        "image")
            elif isinstance(out, PhaseArray):
                out_ = (out, 
                        dict(scale=scale, name=name, colormap="hsv", translate=input.translate,
                                contrast_limits=out.border), 
                        "image")
            elif isinstance(out, Label):
                out_ = (out, 
                        dict(opacity=0.3, scale=scale, name=name), 
                        "labels")
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
    # run_func.setVisible(False)
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
        
# class nDFloatLineEdit(magicgui.widgets.LineEdit):
#     def bind(self, value, call: bool = True) -> None:
#         self._call_bound = call
#         val = eval(value)
#         if isinstance(val, (float, list)):
#             self._bound_value = val
#         else:
#             raise TypeError(val)