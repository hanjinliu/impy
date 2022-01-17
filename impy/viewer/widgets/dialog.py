from __future__ import annotations
import datetime
from qtpy.QtWidgets import (QDialog, QPushButton, QLabel, QGridLayout, QCheckBox, QLineEdit, 
                            QComboBox, QVBoxLayout, QHBoxLayout, QSpinBox, QWidget)
import napari
import numpy as np
from functools import wraps

from ..utils import add_labeledarray, copy_layer, front_image, add_labels, layer_to_impy_object
from ..._const import SetConst
from ...utils.slicer import axis_targeted_slicing
from ...utils.axesop import find_first_appeared

def close_anyway(func):
    @wraps(func)
    def wrapped_func(self:QDialog, *args, **kwargs):
        from napari.utils.notifications import Notification, notification_manager
        try:
            out = func(self, *args, **kwargs)
        except Exception as e:
            notification_manager.dispatch(Notification.from_exception(e))
            out = None
        finally:
            self.close()
        return out
    return wrapped_func

class RegionPropsDialog(QDialog):
    history = "mean_intensity"
    
    def __init__(self, viewer:"napari.Viewer"):
        self.viewer = viewer
        super().__init__(viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self._add_widgets()
    
    @close_anyway
    def run(self, *args):
        selected = list(self.viewer.layers.selection)
        if not any(isinstance(layer, napari.layers.Image) for layer in selected):
            selected = [front_image(self.viewer)]
        
        properties = ("label",) + tuple(self.line.text().split(","))
        
        for layer in selected:
            if not isinstance(layer, napari.layers.Image):
                continue
            
            if not hasattr(layer.data, "labels"):
                raise ValueError(f"Image of layer {layer.name} does not have labels.")
            
            lbl = layer.data.labels
            with SetConst("SHOW_PROGRESS", False):
                out = layer.data.regionprops(properties=properties)
        
            out["label"] = out["label"].astype(lbl.dtype)
            order = np.argsort(out["label"].value)
            prop = {k: np.concatenate([[np.nan], out[k].value[order]]) for k in properties}
            # find Labels layer
            for l in self.viewer.layers:
                if l.metadata.get("destination_image", None) is layer.data:
                    l.properties = prop
                    break
            else:
                l = add_labels(self.viewer, lbl, translate=layer.translate)[0]
                l.properties = prop
            
            from .table import TableWidget
            import pandas as pd
            df = pd.DataFrame(l.properties)
            df = df.drop(df.index[0]) # The first row is background
            table = TableWidget(self.viewer, df, name=f"Properties of {layer.name}")
            self.viewer.window.add_dock_widget(table, area="right", name=table.name)
        
        self.__class__.history = self.line.text()
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
    """
    This dialog is opened when an image layer is duplicated.
    """
    def __init__(self, viewer:"napari.Viewer", layer):
        self.viewer = viewer
        self.layer = layer
        super().__init__(viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self._add_widgets()
        
    @close_anyway
    def run(self, *args):
        line = self.line.text()
        if line.strip() == "" and not self.check.isChecked():
            new_layer = copy_layer(self.layer)
        elif line.strip():
            new_layer = self.duplicate_sliced_layer(self.layer)
        else:
            new_layer = self.duplicate_current_step(self.layer)
        
        self.viewer.add_layer(new_layer)
        return None
    
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
        sl = axis_targeted_slicing(layer.data.ndim, layer.data.axes, key)
        
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

class ProjectionDialog(QDialog):
    def __init__(self, viewer:"napari.Viewer", layer):
        self.viewer = viewer
        self.layer = layer
        self.data = layer_to_impy_object(viewer, layer)
        super().__init__(viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QHBoxLayout())
        self._add_widgets()
    
    def _add_widgets(self): ...

class ImageProjectionDialog(ProjectionDialog):
    """
    This dialog is opened when an image layer is projected.
    """
    @close_anyway
    def run(self, *args):
        img = self.data
        out = img.proj(axis=self.axis.currentText(), method=self.method.currentText())
        translate = [t for a, t in zip(img.axes, self.layer.translate) if a in out.axes]
        add_labeledarray(self.viewer, out, translate=translate, name=f"[Proj]{self.layer.name}")
        return None
    
    def _add_widgets(self):
        self.method = QComboBox(self)
        self.method.setToolTip("Projection method")
        self.method.addItems(["mean", "max", "median", "min", "std"])
        self.method.setCurrentText("mean")
        self.layout().addWidget(self.method)
        
        axes = str(self.data.axes)
        self.axis = QComboBox(self)
        self.axis.setToolTip("Projection axis")
        self.axis.addItems(list(axes[:-2]))
        self.axis.setCurrentText(find_first_appeared("ztpi<c", axes, "yx"))
        self.layout().addWidget(self.axis)
        
        self.run_button= QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)
        self.layout().addWidget(self.run_button)

class LabelProjectionDialog(ProjectionDialog):
    """
    This dialog is opened when an image layer is projected.
    """    
    @close_anyway
    def run(self, *args):
        lbl = self.data
        out = lbl.proj(axis=self.axis.currentText())
        translate = [t for a, t in zip(lbl.axes, self.layer.translate) if a in out.axes]
        add_labels(self.viewer, out, translate=translate, name=f"[Proj]{self.layer.name}")
        return None
    
    def _add_widgets(self):        
        axes = str(self.data.axes)
        self.axis = QComboBox(self)
        self.axis.setToolTip("Projection axis")
        self.axis.addItems(list(axes[:-2]))
        self.axis.setCurrentText(find_first_appeared("ztpi<c", axes, "yx"))
        self.layout().addWidget(self.axis)
        
        self.run_button= QPushButton("Run", self)
        self.run_button.clicked.connect(self.run)
        self.layout().addWidget(self.run_button)
    
class TimeStamper(QDialog):
    def __init__(self, viewer:"napari.Viewer", layer):
        self.viewer = viewer
        self.layer = layer
        super().__init__(viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QVBoxLayout())
        self._add_widgets()
    
    @close_anyway
    def run(self, *args):
        i = np.arange(self.layer.data.shape.t)
        factor = {"hr": 3600, "min": 60, "sec": 1, "msec":0.001}[self.unit.currentText()]
        sec = i * self.dt.value() * factor
        time_stamp = [str(datetime.timedelta(seconds=float(s))) for s in sec]
        taxis = self.layer.data.axisof("t")
        arr_basic = np.zeros((4, self.viewer.dims.ndim))
        arr_basic[1, -2] = 1
        arr_basic[2, -2] = 1
        arr_basic[2, -1] = 1
        arr_basic[3, -1] = 1
        def _rectangle(t):
            arr = arr_basic.copy()
            arr[:, taxis] = t
            return arr
        shapes = [_rectangle(t) for t in i]
        
        text_params = {"text": "{time}",
                       "color": "white",
                       "anchor": "upper_left",
                       "size": 10}
        self.viewer.add_shapes(shapes, text=text_params, properties={"time": time_stamp}, face_color=[0,0,0,0],
                               edge_color=[0,0,0,0], name="Time Stamp")
        return None
    
    def _add_widgets(self):
        wid = QWidget(self)
        wid.setLayout(QHBoxLayout())
        
        self.dt = QSpinBox(wid)
        self.dt.setToolTip("Time per frame")
        self.dt.setValue(10)
        wid.layout().addWidget(self.dt)
        
        self.unit = QComboBox(wid)
        self.unit.setToolTip("Time unit")
        self.unit.addItems(["hr", "min", "sec", "msec"])
        self.unit.setCurrentText("sec")
        wid.layout().addWidget(self.unit)
        
        self.layout().addWidget(wid)
        
        self.run_button= QPushButton("OK", self)
        self.run_button.clicked.connect(self.run)
        self.layout().addWidget(self.run_button)