from __future__ import annotations
import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction

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
           "add_controller_widget", 
           "add_note_widget",
           "edit_properties",
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

def add_imread_menu(viewer:"napari.Viewer"):
    from ..core import imread
    action = QAction("imread ...", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window.qt_viewer,
            caption="Select file ...",
            directory=hist[0],
        )
        if filenames != [] and filenames is not None:
            img = imread(filenames[0])
            add_labeledarray(viewer, img)
            napari.utils.history.update_open_history(filenames[0])
        return None
    
    viewer.window.file_menu.addAction(action)
    return None


def add_imsave_menu(viewer:"napari.Viewer"):
    action = QAction("imsave ...", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        dlg = QFileDialog()
        layers = list(viewer.layers.selection)
        if len(layers) == 0:
            raise ValueError("Select a layer first.")
        elif len(layers) > 1:
            raise ValueError("Select only one layer.")

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
    
    viewer.window.file_menu.addAction(action)
    return None

def add_read_csv_menu(viewer:"napari.Viewer"):
    action = QAction("pandas.read_csv ...", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window.qt_viewer,
            caption="Select file ...",
            directory=hist[0],
        )
        if (filenames != []) and (filenames is not None):
            path = filenames[0]
            read_csv(viewer, path)
            napari.utils.history.update_open_history(filenames[0])
        return None
    
    viewer.window.file_menu.addAction(action)
    return None

def add_explorer_menu(viewer:"napari.Viewer"):
    action = QAction("Open explorer", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        from .widgets import Explorer
        name = "Explorer"
        if name in viewer.window._dock_widgets.keys():
            viewer.window._dock_widgets[name].show()
        else:
            root = napari.utils.history.get_open_history()[0]
            ex = Explorer(viewer, root)
            viewer.window.add_dock_widget(ex, name="Explorer", area="right", allowed_areas=["right"])
        return None
    action.setShortcut("Ctrl+Shift+E")
    viewer.window.file_menu.addAction(action)
    return None


def add_duplicate_menu(viewer:"napari.Viewer"):
    action = QAction("Duplicate", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        """
        Duplicate the selected layer.
        """
        layer = get_a_selected_layer(viewer)
        
        if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
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
        if isinstance(layer, napari.layers.Image):
            if layer.data.ndim < 3:
                return None
            dlg = ImageProjectionDialog(viewer, layer)
            dlg.exec_()
            
        elif isinstance(layer, napari.layers.Labels):
            if layer.data.ndim < 3:
                return None
            dlg = LabelProjectionDialog(viewer, layer)
            dlg.exec_()
                
        elif isinstance(layer, napari.layers.Shapes):
            data = [d[:,-2:] for d in data]
            
            if layer.nshapes > 0:
                for k in ["face_color", "edge_color", "edge_width"]:
                    kwargs[k] = getattr(layer, k)
            else:
                data = None
            kwargs["ndim"] = 2
            kwargs["shape_type"] = layer.shape_type
            viewer.add_shapes(data, **kwargs)
            
        elif isinstance(layer, napari.layers.Points):
            data = data[:, -2:]
            for k in ["face_color", "edge_color", "size", "symbol"]:
                kwargs[k] = getattr(layer, k, None)
            kwargs["size"] = layer.size[:,-2:]
            if len(data) == 0:
                data = None
            kwargs["ndim"] = 2
            viewer.add_points(data, **kwargs)
            
        elif isinstance(layer, napari.layers.Tracks):
            data = data[:, [0, -2, -1]] # p, y, x axes
            viewer.add_tracks(data, **kwargs)
            
        else:
            raise NotImplementedError(type(layer))
    
    action.setShortcut("Ctrl+P")
    viewer.window.layer_menu.addAction(action)
    return None

def add_crop_menu(viewer:"napari.Viewer"):
    action = QAction("Crop", viewer.window._qt_window)
    @action.triggered.connect
    def _():
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
                crop_func = _crop_rectangle
            else:
                crop_func = _crop_rotated_rectangle
            
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
    @action.triggered.connect
    def _():
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
                print(f"Label already exist in {dst}. Overlapped.")
                del dst.labels
            dst.append_label(labelout)
            
        return None
    
    viewer.window.layer_menu.addAction(action)
    return None

def add_time_stamper_menu(viewer:"napari.Viewer"):
    action = QAction("Add time stamp", viewer.window._qt_window)
    @action.triggered.connect
    def _():
        layer = get_a_selected_layer(viewer)
        if not isinstance(layer, napari.layers.Image):
            raise TypeError("Select an image layer.")
        dlg = TimeStamper(viewer, layer)
        dlg.exec_()
        return None
    
    viewer.window.layer_menu.addAction(action)
    return None

def edit_properties(viewer:"napari.Viewer"):
    """
    Edit properties of selected shapes or points.
    """    
    line = magicgui.widgets.LineEdit(tooltip="Property editor")
    @line.changed.connect
    def _(event):
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
        
    viewer.window.add_dock_widget(line, area="left", name="Property editor")
    return None

def add_controller_widget(viewer:"napari.Viewer"):
    controller_widget = Controller(viewer)
    viewer.window.add_dock_widget(controller_widget, area="left", name="impy controller")
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
    def _():
        @magicgui.magicgui(call_button="Match",
                           img={"label": "image",
                                "tooltip": "Reference image. This image will not move."},
                           template={"label": "temp",
                                     "tooltip": "Template image. This image will move."},
                           ndim={"choices": [2, 3]})
        def template_matcher(img:napari.layers.Image, template:napari.layers.Image, ndim=2):
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
    def _():
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

def add_time_stamper(viewer:"napari.Viewer"):
    ...

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


def _crop_rotated_rectangle(img, crds, dims):
    translate = np.min(crds, axis=0)
    
    # check is sorted
    ids = [img.axisof(a) for a in dims]
    if sorted(ids) == ids:
        cropped_img = img.rotated_crop(crds[1], crds[0], crds[2], dims=dims)
    else:
        crds = np.fliplr(crds)
        cropped_img = img.rotated_crop(crds[3], crds[0], crds[2], dims=dims)
    
    return cropped_img, translate

def _crop_rectangle(img, crds, dims):
    start = crds[0]
    end = crds[2]
    sl = []
    translate = np.empty(2)
    for i in [0, 1]:
        sl0 = sorted([start[i], end[i]])
        x0 = max(int(sl0[0]), 0)
        x1 = min(int(sl0[1]), img.sizeof(dims[i]))
        sl.append(f"{dims[i]}={x0}:{x1}")
        translate[i] = x0
    
    area_to_crop = ";".join(sl)
    
    cropped_img = img[area_to_crop]
    return cropped_img, translate

def _get_property(layer, i):
    try:
        prop = layer.properties["text"][i]
    except (KeyError, IndexError):
        prop = ""
    if prop != "":
        prop = prop + " of "
    return prop