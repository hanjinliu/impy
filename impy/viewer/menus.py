from __future__ import annotations
from warnings import warn
import napari
from napari.utils.notifications import Notification, notification_manager
from napari.utils import history
from napari.layers import Image, Shapes, Points, Labels, Tracks
import magicgui
from functools import wraps
from qtpy.QtWidgets import QFileDialog, QAction
from qtpy.QtGui import QCursor

from .widgets import *
from .widgets.table import read_csv
from .utils import *
from .._const import SetConst

__all__ = ["add_imread_menu",
           "add_imsave_menu",
           "add_read_csv_menu",
           "add_explorer_menu",
           "add_duplicate_menu",
           "add_proj_menu",
           "add_crop_menu",
           "add_layer_to_labels_menu",
           "add_time_stamper_menu",
           "add_text_layer_menu",
           "add_get_props_menu",
           "add_label_menu",
           "add_plane_clipper",
           "add_note_widget",
           "add_threshold", 
           "add_rotator",
           "add_filter", 
           "add_regionprops",
           "add_rectangle_editor",
           "layer_template_matcher",
           "function_handler",
           ]


FILTERS = ["None", "gaussian_filter", "median_filter", "mean_filter", "dog_filter", "doh_filter", "log_filter", 
           "erosion", "dilation", "opening", "closing", "entropy_filter", "std_filter", "coef_filter",
           "tophat", "rolling_ball"]

def catch_notification(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        try:
            out = func(*args, **kwargs)
        except Exception as e:
            out = None
            notification_manager.dispatch(Notification.from_exception(e))
        return out
    return wrapped

def add_imread_menu(viewer:"napari.Viewer"):
    from ..core import imread
    action = QAction("imread ...", viewer.window._qt_window)
    @action.triggered.connect
    @catch_notification
    def _(*args):
        dlg = QFileDialog()
        hist = history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window._qt_window,
            caption="Select file ...",
            directory=hist[0],
        )
        if filenames != [] and filenames is not None:
            img = imread(filenames[0])
            add_labeledarray(viewer, img)
            history.update_open_history(filenames[0])
        return None
    
    viewer.window.file_menu.addAction(action)
    return None


def add_imsave_menu(viewer:"napari.Viewer"):
    action = QAction("imsave ...", viewer.window._qt_window)
    @action.triggered.connect
    @catch_notification
    def _(*args):
        dlg = QFileDialog()
        layers = list(viewer.layers.selection)
        if len(layers) == 0:
            raise ValueError("Select a layer first.")
        elif len(layers) > 1:
            raise ValueError("Select only one layer.")

        img = layer_to_impy_object(viewer, layers[0])
        hist = history.get_save_history()
        dlg.setHistory(hist)
        
        if img.dirpath:
            last_hist = img.dirpath
        else:
            last_hist = hist[0]
        filename, _ = dlg.getSaveFileName(
            parent=viewer.window._qt_window,
            caption="Save image layer",
            directory=last_hist,
        )

        if filename:
            img.imsave(filename)
            history.update_save_history(filename)
        return None
    
    viewer.window.file_menu.addAction(action)
    return None

def add_read_csv_menu(viewer:"napari.Viewer"):
    action = QAction("pandas.read_csv ...", viewer.window._qt_window)
    @action.triggered.connect
    @catch_notification
    def _(*args):
        dlg = QFileDialog()
        hist = history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window._qt_window,
            caption="Select file ...",
            directory=hist[0],
        )
        if (filenames != []) and (filenames is not None):
            path = filenames[0]
            read_csv(viewer, path)
            history.update_open_history(filenames[0])
        return None
    
    viewer.window.file_menu.addAction(action)
    return None

def add_explorer_menu(viewer:"napari.Viewer"):
    action = QAction("Open explorer", viewer.window._qt_window)
    action.setStatusTip("Add an file tree widget in the viewer. You can open, filter, copy or see the summary of files under a root directory.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        from .widgets import Explorer
        name = "Explorer"
        if name in viewer.window._dock_widgets.keys():
            viewer.window._dock_widgets[name].show()
        else:
            root = history.get_open_history()[0]
            ex = Explorer(viewer, root)
            viewer.window.add_dock_widget(ex, name="Explorer", area="right", allowed_areas=["right"])
        return None
    action.setShortcut("Ctrl+Shift+E")
    viewer.window.file_menu.addAction(action)
    return None


def add_duplicate_menu(viewer:"napari.Viewer"):
    action = QAction("Duplicate", viewer.window._qt_window)
    action.setStatusTip("Duplicate the selected layer. If an image or labels layer is selected, "
                        "you can set precise duplication method in a dialog.")
    @action.triggered.connect
    def _():
        layer = get_a_selected_layer(viewer)
        
        if isinstance(layer, (Image, Labels)):
            dlg = DuplicateDialog(viewer, layer)
            dlg.exec_()
        else:
            new_layer = copy_layer(layer)
            viewer.add_layer(new_layer)
        return None    

    action.setShortcut("Ctrl+Shift+D")
    viewer.window.layer_menu.addAction(action)
    return None

def add_proj_menu(viewer:"napari.Viewer"):
    action = QAction("Project", viewer.window._qt_window)
    action.setStatusTip("Project the selected layer. If an image or labels layer is selected, "
                        "you can set precise projection method in a dialog.")
    @action.triggered.connect
    def _():
        """
        Projection
        """
        layer = get_a_selected_layer(viewer)
        
        data = layer.data
        kwargs = {}
        kwargs.update({"scale": layer.scale[-2:], 
                       "translate": layer.translate[-2:],
                       "blending": layer.blending,
                       "opacity": layer.opacity,
                       "ndim": 2,
                       "name": f"[Proj]{layer.name}"})
        if isinstance(layer, Image):
            if layer.data.ndim < 3:
                return None
            dlg = ImageProjectionDialog(viewer, layer)
            dlg.exec_()
            
        elif isinstance(layer, Labels):
            if layer.data.ndim < 3:
                return None
            dlg = LabelProjectionDialog(viewer, layer)
            dlg.exec_()
                
        elif isinstance(layer, Shapes):
            data = [d[:,-2:] for d in data]
            
            if layer.nshapes > 0:
                for k in ["face_color", "edge_color", "edge_width"]:
                    kwargs[k] = getattr(layer, k)
            else:
                data = None
            kwargs["ndim"] = 2
            kwargs["shape_type"] = layer.shape_type
            viewer.add_shapes(data, **kwargs)
            
        elif isinstance(layer, Points):
            data = data[:, -2:]
            for k in ["face_color", "edge_color", "size", "symbol"]:
                kwargs[k] = getattr(layer, k, None)
            kwargs["size"] = layer.size[:,-2:]
            if len(data) == 0:
                data = None
            kwargs["ndim"] = 2
            viewer.add_points(data, **kwargs)
            
        elif isinstance(layer, Tracks):
            data = data[:, [0, -2, -1]] # p, y, x axes
            viewer.add_tracks(data, **kwargs)
            
        else:
            raise NotImplementedError(type(layer))
    
    action.setShortcut("Ctrl+P")
    viewer.window.layer_menu.addAction(action)
    return None

def add_crop_menu(viewer:"napari.Viewer"):
    action = QAction("Crop", viewer.window._qt_window)
    action.setStatusTip("Crop selected image layers with rectangle shapes.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        """
        Crop images with (rotated) rectangle shapes.
        """        
        from ..core import array as ip_array
        if viewer.dims.ndisplay == 3:
            viewer.status = "Cannot crop in 3D mode."
        imglist = list(iter_selected_layer(viewer, "Image"))
        if len(imglist) == 0:
            imglist = [front_image(viewer)]
        
        ndim = np.unique([shape_layer.ndim for shape_layer 
                        in iter_selected_layer(viewer, "Shapes")])
        if len(ndim) > 1:
            viewer.status = "Cannot crop using Shapes layers with different number of dimensions."
        else:
            ndim = ndim[0]
        
        if ndim == viewer.dims.ndim:
            active_plane = list(viewer.dims.order[-2:])
        else:
            active_plane = [-2, -1]
        dims2d = "".join(viewer.dims.axis_labels[i] for i in active_plane)
        rects = []
        for shape_layer in iter_selected_layer(viewer, "Shapes"):
            for i in range(shape_layer.nshapes):
                shape = shape_layer.data[i]
                type_ = shape_layer.shape_type[i]
                if type_ == "rectangle":
                    rects.append((shape[:, active_plane],            # shape = float pixel
                                shape_layer.scale[active_plane],
                                _get_property(shape_layer, i))
                                )

        for rect, shape_layer_scale, prop in rects:
            if np.any(np.abs(rect[0] - rect[1])<1e-5):
                crop_func = crop_rectangle
            else:
                crop_func = crop_rotated_rectangle
            
            for layer in imglist:
                factor = layer.scale[active_plane]/shape_layer_scale
                _name = prop + layer.name
                layer = viewer.add_layer(copy_layer(layer))
                dr = layer.translate[active_plane] / layer.scale[active_plane]
                newdata, relative_translate = \
                    crop_func(layer.data, rect/factor - dr, dims2d)
                if newdata.size <= 0:
                    continue
                
                if not isinstance(newdata, (LabeledArray, LazyImgArray)):
                    scale = get_viewer_scale(viewer)
                    axes = "".join(viewer.dims.axis_labels)
                    newdata = ip_array(newdata, axes=axes)
                    newdata.set_scale(**scale)
                    
                newdata.dirpath = layer.data.dirpath
                newdata.metadata = layer.data.metadata
                newdata.name = layer.data.name
                    
                layer.data = newdata
                translate = layer.translate
                translate[active_plane] += relative_translate * layer.scale[active_plane]
                layer.translate = translate
                layer.name = _name
                layer.metadata.update({"init_translate": layer.translate, 
                                    "init_scale": layer.scale})
                
        # remove original images
        [viewer.layers.remove(img) for img in imglist]
        return None
    
    action.setShortcut("Ctrl+Shift+X")
    viewer.window.layer_menu.addAction(action)
    return None

def add_layer_to_labels_menu(viewer:"napari.Viewer"):   
    action = QAction("Layer to labels", viewer.window._qt_window)
    action.setStatusTip("Convert the selected shapes layer to a labels layer.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        """
        Convert manually drawn shapes to labels and store it.
        """        
        # determine destinations.
        destinations = [l.data for l in iter_selected_layer(viewer, "Image")]
        if len(destinations) == 0:
            destinations = [front_image(viewer).data]
        
        for dst in destinations:
            # check zoom_factors
            d = viewer.dims
            scale = {a: r[2] for a, r in zip(d.axis_labels, d.range)}
            # zoom_factors = layer.scale[-2:]/shape_layer_scale[-2:]
            zoom_factors = [scale[a]/dst.scale[a] for a in "yx"]
            if np.unique(zoom_factors).size == 1:
                zoom_factor = zoom_factors[0]
            else:
                raise ValueError("Scale mismatch in images and napari world.")
            
            # make labels from selected layers
            shapes = [to_labels(layer, dst.shape, zoom_factor=zoom_factor) 
                    for layer in iter_selected_layer(viewer, "Shapes")]
            labels = [layer.data for layer in iter_selected_layer(viewer, "Labels")]
            
            if len(shapes) > 0 and len(labels) > 0:
                viewer.status = "Both Shapes and Labels were selected"
                return None
            elif len(shapes) > 0:
                labelout = np.max(shapes, axis=0)
                viewer.add_labels(labelout, opacity=0.3, scale=list(scale.values()), name="Labeled by Shapes")
            elif len(labels) > 0:
                labelout = np.max(labels, axis=0)
            else:
                return None
            
            # append labels to each destination
            if hasattr(dst, "labels"):
                warn(f"Label already exist in {dst}. Overlapped.", UserWarning)
                del dst.labels
            dst.append_label(labelout)
            
        return None
    
    viewer.window.layer_menu.addAction(action)
    return None

def add_time_stamper_menu(viewer:"napari.Viewer"):
    action = QAction("Add time stamp", viewer.window._qt_window)
    action.setStatusTip("Add a shapes layer with time stamp texts.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        layer = get_a_selected_layer(viewer)
        if not isinstance(layer, Image):
            raise TypeError("Select an image layer.")
        layer.data.axisof("t") # check image axes here.
        dlg = TimeStamper(viewer, layer)
        dlg.exec_()
        return None
    
    viewer.window.layer_menu.addAction(action)
    return None

def add_get_props_menu(viewer:"napari.Viewer"):
    action = QAction("Open property table", viewer.window._qt_window)
    action.setStatusTip("Get properties from selected layer(s) and show in table widget(s).")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        layers = list(iter_selected_layer(viewer, ["Points", "Tracks", "Shapes", "Labels"]))
        if len(layers) == 0:
            raise ValueError("No Points, Tracks or Shapes layer selected")
        
        for layer in layers:
            name = f"Properties of {layer.name}"
            ndata = len(layer.data)
            if layer.properties == {} and ndata > 0:
                layer.current_properties = {"1": np.array([""], dtype="<U32")}
                layer.properties = {"1": np.array([""]*ndata, dtype="<U32")}
            widget = TableWidget(viewer, layer.properties, name=name)
            if isinstance(layer, (Points, Shapes)):
                widget.linked_layer = layer
            viewer.window.add_dock_widget(widget, area="right", name=name)
            
        return None
    
    viewer.window.layer_menu.addAction(action)
    return None

def add_text_layer_menu(viewer:"napari.Viewer"):
    action = QAction("Add a text layer", viewer.window._qt_window)
    action.setStatusTip("Add a shapes layer as a text layer. Shapes themselves are invisible. Use property editor to edit their names.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        layer = viewer.add_shapes(ndim=2, 
                                  shape_type="rectangle",
                                  name="Text Layer",
                                  properties={"text": np.array(["text here"], dtype="<U32")},
                                  blending = "additive",
                                  edge_width=2.0,
                                  face_color=[0,0,0,0],
                                  edge_color=[0,0,0,0],
                                  text={"text": "{text}", 
                                        "size": 6.0 * Const["FONT_SIZE_FACTOR"],
                                        "color": "white",
                                        "anchor": "center"}
                                  )
        layer.mode = "add_rectangle"
        layer._rotation_handle_length = 20/np.mean(layer.scale[-2:])
        
        @layer.mouse_double_click_callbacks.append
        def edit(layer, event):
            i, _ = layer.get_value(
                event.position,
                view_direction=event.view_direction,
                dims_displayed=event.dims_displayed,
                world=True
            )

            if i is None:
                return None
            
            window_geo = viewer.window._qt_window.geometry()
            pos = QCursor().pos()
            x = pos.x() - window_geo.x()
            y = pos.y() - window_geo.y()
            line = QLineEdit(viewer.window._qt_window)
            edit_geometry = line.geometry()
            edit_geometry.setWidth(140)
            edit_geometry.moveLeft(x)
            edit_geometry.moveTop(y)
            line.setGeometry(edit_geometry)
            f = line.font()
            f.setPointSize(20)
            line.setFont(f)
            line.setText(layer.text.values[i])
            line.setHidden(False)
            line.setFocus()
            line.selectAll()
            @line.textChanged.connect
            def _():
                old = layer.properties.get("text", [""]*len(layer.data))
                old[i] = line.text().strip()
                layer.text.refresh_text({"text": old})
                return None
            @line.editingFinished.connect
            def _():
                line.setHidden(True)
                line.deleteLater()
                return None
            
    viewer.window.layer_menu.addAction(action)
    return None


def add_label_menu(viewer:"napari.Viewer"):
    action = QAction("Label ImgArray", viewer.window._qt_window)
    action.setStatusTip("Add labels layer that is connected with the selected image layer.")
    @action.triggered.connect
    @catch_notification
    def _(*args):
        selected = list(viewer.layers.selection)
        if len(selected) != 1:
            raise ValueError("No layer selected")
        selected = selected[0]
        if not isinstance(selected, Image):
            raise TypeError("Selected layer is not an image layer")
        img = selected.data
        if hasattr(img, "labels"):
            raise ValueError("Image layer already has labels.")
        with SetConst("SHOW_PROGRESS", False):
            img.append_label(np.zeros(img.shape, dtype=np.uint8))
            
        layer = viewer.add_labels(img.labels.value, opacity=0.3, scale=selected.scale, 
                                        name=f"[L]{img.name}", translate=selected.translate,
                                        metadata={"destination_image": img})
        layer.mode = "paint"
        return None

    viewer.window.layer_menu.addAction(action)
    return None

def add_plane_clipper(viewer:"napari.Viewer"):
    action = QAction("Clipping plane", viewer.window._qt_window)
    @action.triggered.connect
    def _(*args):
        layer = get_a_selected_layer(viewer)
        
        wid = PlaneClipRange(viewer)
        wid.connectLayer(layer)
        viewer.window.add_dock_widget(wid, name="Plane Clip", area="left")
        
        return None
    viewer.window.layer_menu.addAction(action)
    return None
    
def add_note_widget(viewer:"napari.Viewer"):
    text = magicgui.widgets.TextEdit(tooltip="Note")
    text = viewer.window.add_dock_widget(text, area="right", name="Note")
    text.setVisible(False)
    return None

def add_gui_to_function_menu(viewer:"napari.Viewer", gui:type, name:str):
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

def add_filter(viewer:"napari.Viewer"):
    return add_gui_to_function_menu(viewer, FunctionCaller, "Filters")

def add_threshold(viewer:"napari.Viewer"):
    return add_gui_to_function_menu(viewer, ThresholdAndLabel, "Threshold/Label")

def add_rectangle_editor(viewer:"napari.Viewer"):
    return add_gui_to_function_menu(viewer, RectangleEditor, "Rectangle Editor")

def add_rotator(viewer:"napari.Viewer"):
    return add_gui_to_function_menu(viewer, Rotator, "Rotation")

def add_regionprops(viewer:"napari.Viewer"):
    action = QAction("Measure Region Properties", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        dlg = RegionPropsDialog(viewer)
        dlg.exec_()
        
    viewer.window.function_menu.addAction(action)
    
    return None


def layer_template_matcher(viewer:"napari.Viewer"):
    action = QAction("Template Matcher", viewer.window._qt_window)
    @action.triggered.connect
    @catch_notification
    def _(*args):
        @magicgui.magicgui(call_button="Match",
                           img={"label": "image",
                                "tooltip": "Reference image. This image will not move."},
                           template={"label": "temp",
                                     "tooltip": "Template image. This image will move."},
                           ndim={"choices": [2, 3]})
        def template_matcher(img:Image, template:Image, ndim=2):
            step = viewer.dims.current_step[:-min(ndim, img.ndim)]
            img_ = img.data[step]
            template_ = template.data[step]
            with SetConst("SHOW_PROGRESS", False):
                res = img_.ncc_filter(template_)
            maxima = np.unravel_index(np.argmax(res), res.shape)
            maxima = tuple((m - l//2)*s for m, l, s in zip(maxima, template_.shape, template.scale))
            template.translate = img.translate + np.array(step + maxima)
    
        viewer.window.add_dock_widget(template_matcher, area="left", name="Template Matcher")
        return None
    
    viewer.window.function_menu.addAction(action)
    return None   

def function_handler(viewer:"napari.Viewer"):
    action = QAction("Function Handler", viewer.window._qt_window)
    @action.triggered.connect
    @catch_notification
    def _(*args):
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
    
    viewer.window.function_menu.addAction(action)
    return None


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



def _get_property(layer:Shapes|Points, i):
    try:
        prop = layer.properties["text"][i]
    except (KeyError, IndexError):
        prop = ""
    if prop != "":
        prop = prop + " of "
    return prop