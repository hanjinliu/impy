from magicgui.widgets import FunctionGui
import napari
import inspect
import numpy as np
from ..utils import image_tuple, label_tuple

from ..._const import SetConst


RANGES = {"None": (None, None), 
          "gaussian_filter": (0.2, 30),
          "median_filter": (1, 30),
          "mean_filter": (1, 30),
          "lowpass_filter": (0.005, 0.5),
          "highpass_filter": (0.005, 0.5),
          "erosion": (1, 30), 
          "dilation": (1, 30), 
          "opening": (1, 30), 
          "closing": (1, 30),
          "tophat": (5, 30), 
          "entropy_filter": (1, 30), 
          "enhance_contrast": (1, 30),
          "std_filter": (1, 30), 
          "coef_filter": (1, 30),
          "dog_filter": (0.2, 30),
          "doh_filter": (0.2, 30),
          "log_filter": (0.2, 30), 
          "rolling_ball": (5, 30),
          }

class FunctionCaller(FunctionGui):
    def __init__(self, viewer):
        self.viewer = viewer            # parent napari viewer object
        self.running_function = None    # currently running function
        self.current_layer = None       # currently selected layer
        self.last_inputs = None         # last inputs including function name
        self.last_outputs = None        # last output from the function

        opt = dict(funcname={"choices": list(RANGES.keys()), "label": "function"},
                   param={"widget_type": "FloatSlider", "min":0.01, "max": 30,
                          "tooltip": "The first parameter."},
                   dims={"choices": ["2D", "3D"], "tooltip": "Spatial dimensions"},
                   fix_clims={"widget_type": "CheckBox", "label": "fix contrast limits",
                              "tooltip": "If you'd like to fix the contrast limits\n"
                                         "while parameter sweeping, check here"}
                   )
    
        def _func(layer:napari.layers.Image, funcname:str, param, dims="2D", 
                  fix_clims=False) -> napari.types.LayerDataTuple:
            self.current_layer = layer

            if layer is None or funcname == "None" or not self.visible:
                return None

            name = f"Result of {layer.name}"
            inputs =  (layer.name, funcname, param, dims)

            # run function if needed
            if self.last_inputs == inputs:
                pass
            else:
                try:
                    with SetConst("SHOW_PROGRESS", False):
                        self.last_outputs = self.running_function(param, dims=int(dims[0]))
                except Exception as e:
                    self.viewer.status = f"{funcname} finished with {e.__class__.__name__}: {e}"
                    return None
                else:
                    self.last_inputs = inputs
            # set the parameters for the output layer
            try:
                if fix_clims:
                    props_to_inherit = ["colormap", "blending", "translate", "scale", "contrast_limits"]
                else:
                    props_to_inherit = ["colormap", "blending", "translate", "scale"]
                kwargs = {k: getattr(self.viewer.layers[name], k, None) for k in props_to_inherit}
            except KeyError:
                kwargs = dict(translate="inherit")
                
            return image_tuple(layer, self.last_outputs, name=name, **kwargs)
        
        super().__init__(_func, auto_call=True, param_options=opt)
        self.funcname.changed.connect(self.update_widget)

    def update_widget(self, event=None):
        """
        Update the widget labels and sliders every time function is changed.
        """
        name = self.funcname.value
        self.running_function = getattr(self.current_layer.data, name, None)
        if name == "None" or self.running_function is None:
            return None
        pmin, pmax = RANGES[name]
        sig = inspect.signature(self.running_function)
        first_param = list(sig.parameters.keys())[0]
        self.param.label = first_param
        self.param.min = pmin
        self.param.max = pmax
        self.param.value = sig.parameters[first_param].default
        return None
        
class ThresholdAndLabel(FunctionGui):
    def __init__(self, viewer):
        self.viewer = viewer
        opt = dict(percentile={"widget_type": "FloatSlider", 
                               "min": 0, "max": 100,
                               "tooltip": "Threshold percentile"},
                   label={"widget_type": "CheckBox"}
                   )
        def _func(layer:napari.layers.Image, percentile=50, label=False) -> napari.types.LayerDataTuple:
            if not self.visible:
                return None
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
                        _as_layer_data_tuple = label_tuple
                    else:
                        out = layer.data.threshold(thr)
                        props_to_inherit = ["colormap", "opacity", "blending", "translate", "scale"]
                        _as_layer_data_tuple = image_tuple
                try:
                    kwargs = {k: getattr(viewer.layers[name], k, None) for k in props_to_inherit}
                except KeyError:
                    if label:
                        kwargs = dict(translate=layer.translate, opacity=0.3)
                    else:
                        kwargs = dict(translate=layer.translate, colormap="red", blending="additive")
                
                return _as_layer_data_tuple(layer, out, name=name, **kwargs)
            return None
        
        super().__init__(_func, auto_call=True, param_options=opt)

class RectangleEditor(FunctionGui):
    def __init__(self, viewer):
        self.viewer = viewer
        opt = dict(len_v={"widget_type": "SpinBox", 
                          "label": "V",
                          "tooltip": "vertical length in pixel"},
                   len_h={"widget_type": "SpinBox", 
                          "label": "H",
                          "tooltip": "horizontal length in pixel"})

        def _func(len_v=128, len_h=128):
            selected_layer = self.get_selected_shapes_layer()
    
            # check if one shape/point is selected
            new_data = selected_layer.data
            selected_data = selected_layer.selected_data
            count = 0
            for i, data in enumerate(new_data):
                if selected_layer.shape_type[i] == "rectangle" and i in selected_data:
                    dh = data[1, -2:] - data[0, -2:]
                    dv = data[3, -2:] - data[0, -2:]
                    data[1, -2:] = dh / np.hypot(*dh) * len_h + data[0, -2:]
                    data[3, -2:] = dv / np.hypot(*dv) * len_v + data[0, -2:]
                    data[2, -2:] = data[1, -2:] - data[0, -2:] + data[3, -2:]
                    
                    count += 1
            
            if count == 0:
                if selected_layer.nshapes == 0:
                    # TODO: https://github.com/napari/napari/pull/2961
                    # May be solved in near future
                    return None
                data = np.zeros((4, selected_layer.ndim), dtype=np.float64)
                data[:, :-2] = viewer.dims.current_step[:-2]
                data[1, -2:] = np.array([  0.0, len_h])
                data[2, -2:] = np.array([len_v, len_h])
                data[3, -2:] = np.array([len_v,   0.0])
                new_data = selected_layer.data + [data]
                selected_data = {len(new_data) - 1}
                
            selected_layer.data = new_data       
            selected_layer.selected_data = selected_data
            selected_layer._set_highlight()
            
            return None
        
        super().__init__(_func, auto_call=True, param_options=opt)

    def get_selected_shapes_layer(self):
        selected_layer = list(self.viewer.layers.selection)
        if len(selected_layer) != 1:
            return None
        selected_layer = selected_layer[0]
        if not isinstance(selected_layer, napari.layers.Shapes):
            return None
        elif len(selected_layer.selected_data) == 0:
            return None
        return selected_layer