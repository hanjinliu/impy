from qtpy.QtWidgets import QDialog, QPushButton, QLabel, QGridLayout, QCheckBox, QLineEdit
import napari
import numpy as np

from ..utils import copy_layer
from ..._const import SetConst
from ...utils.slicer import axis_targeted_slicing


class RegionPropsDialog(QDialog):
    history = "mean_intensity"
    
    def __init__(self, viewer):
        super().__init__(viewer.window._qt_window)
        self.viewer = viewer
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self._add_widgets()
    
    
    def run(self):
        selected = list(self.viewer.layers.selection)
        if len(selected) < 1:
            return None
        
        properties = ("label",) + tuple(self.line.text().split(","))
        
        for imglayer in selected:
            if not isinstance(imglayer, napari.layers.Image):
                continue
            lbl = imglayer.data.labels
            with SetConst("SHOW_PROGRESS", False):
                out = imglayer.data.regionprops(properties=properties)
            out["label"] = out["label"].astype(lbl.dtype)
            order = np.argsort(out["label"].value)
            prop = {k: np.concatenate([[0], out[k].value[order]]) for k in properties}
            # find Labels layer
            for l in self.viewer.layers:
                if l.metadata.get("destination_image", None) is imglayer.data:
                    l.properties = prop
                    break
            else:
                l = self.viewer.add_labels(lbl.value, opacity=0.3, scale=imglayer.scale, 
                                    name=f"[L]{imglayer.name}", translate=imglayer.translate)
                l.properties = prop
        
        self.__class__.history = self.line.text()
        self.close()
        return None
    
    def _add_widgets(self):
        self.line = QLineEdit(self)
        self.line.setText(self.__class__.history)
        self.layout().addWidget(self.line)
        
        self.run_button= QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)
        self.layout().addWidget(self.run_button)
        return None
    

class DuplicateDialog(QDialog):
    def __init__(self, viewer):
        self.viewer = viewer
        super().__init__(viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self._add_widgets()
        
    
    def run(self):
        line = self.line.text()
        for layer in list(self.viewer.layers.selection):
            if line.strip() == "" and not self.check.isChecked():
                new_layer = copy_layer(layer)
            elif line.strip():
                new_layer = self.duplicate_sliced_layer(layer)
            else:
                new_layer = self.duplicate_current_step(layer)
            
            self.viewer.add_layer(new_layer)
        
        self.close()
            
    def duplicate_current_step(self, layer):
        sl = self.viewer.dims.current_step[:-2]
        
        data, kwargs, *_ = layer.as_layer_data_tuple()
        if isinstance(layer, (napari.layers.Image, napari.layers.Labels)):
            data = data[sl]
        else:
            raise TypeError("Cannot duplicate DataFrame with current step.")
        
        # linear interpolation is valid only in 3D mode.
        if kwargs["interpolation"] == "linear":
            kwargs = kwargs.copy()
            kwargs["interpolation"] = "nearest"
        
        kwargs["scale"] = kwargs["scale"][-2:]
        kwargs["translate"] = kwargs["translate"][-2:]
        kwargs["rotate"] = np.array(kwargs["rotate"])[-2:, -2:]
        kwargs["shear"] = None
        
        copy = layer.__class__(data, **kwargs)
        return copy
        
    def duplicate_sliced_layer(self, layer):
        key = self.line.text().strip("'").strip('"')
        sl = axis_targeted_slicing(layer.data, layer.data.axes, key)
        
        data, kwargs, *_ = layer.as_layer_data_tuple()
        try:
            data = data[key]
        except Exception:
            raise ValueError(f"Cannot duplicate layer with string {key}")
        
        # linear interpolation is valid only in 3D mode.
        if kwargs["interpolation"] == "linear":
            kwargs = kwargs.copy()
            kwargs["interpolation"] = "nearest"
        
        kwargs["scale"] = [a for i, a in enumerate(kwargs["scale"]) if not isinstance(sl[i], int)]
        kwargs["translate"] = [a for i, a in enumerate(kwargs["translate"]) if not isinstance(sl[i], int)]
        kwargs["rotate"] = None
        kwargs["shear"] = None
        
        copy = layer.__class__(data, **kwargs)
        return copy
    
    def _add_widgets(self):        
        label = QLabel(self)
        label.setText('Enter such as "t=1;z=5:8" if needed.')
        self.layout().addWidget(label)
        
        self.line = QLineEdit(self)
        self.layout().addWidget(self.line)
        
        self.check = QCheckBox(self)
        self.check.setText("or duplicate current 2D slice.")
        self.layout().addWidget(self.check)
        
        self.run_button= QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)
        self.layout().addWidget(self.run_button)
        return None
        
    