from __future__ import annotations
from superqt import QRangeSlider
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QLabel
from qtpy.QtCore import Qt
import napari

class PlaneClipRange(QWidget):
    def __init__(self, viewer:"napari.Viewer"):
        super().__init__(viewer.window._qt_window)
        self.xrange = QRangeSlider(Qt.Horizontal, parent=self)
        self.yrange = QRangeSlider(Qt.Horizontal, parent=self)
        self.setLayout(QVBoxLayout())
        
        xframe = QFrame(self)
        xframe.setLayout(QHBoxLayout())
        
        xlabel = QLabel(xframe)
        xlabel.setText("x")
        xframe.layout().addWidget(xlabel)
        xframe.layout().addWidget(self.xrange)
        
        
        yframe = QFrame(self)
        yframe.setLayout(QHBoxLayout())
        
        ylabel = QLabel(yframe)
        ylabel.setText("y")
        yframe.layout().addWidget(ylabel)
        yframe.layout().addWidget(self.yrange)
        
        self.layout().addWidget(xframe)
        self.layout().addWidget(yframe)
        
    
    def connectLayer(self, layer:"napari.components.Layer"):
        xmin = layer.extent.data[0,-1]
        xmax = layer.extent.data[1,-1]
        ymin = layer.extent.data[0,-2]
        ymax = layer.extent.data[1,-2]
        
        self.xrange.setRange(xmin, xmax)
        self.yrange.setRange(ymin, ymax)
        self.xrange.setValue((xmin, xmax))
        self.yrange.setValue((ymin, ymax))
        self.connected_layer = layer
        ndim = layer.ndim
        
        self.connected_layer.experimental_clipping_planes = [
            {"position": (0,)*(ndim-1)+(xmin,), "normal": (0, 0, 1), "enabled": True},
            {"position": (0,)*(ndim-1)+(xmax,), "normal": (0, 0, -1), "enabled": True},
            {"position": (0,)*(ndim-2)+(ymin, 0), "normal": (0, 1, 0), "enabled": True},
            {"position": (0,)*(ndim-2)+(ymax, 0), "normal": (0, -1, 0), "enabled": True},
            ]
        
        self.xrange.valueChanged.connect(self.updateX)
        self.yrange.valueChanged.connect(self.updateY)
    
    @property
    def xminPlane(self):
        return self.connected_layer.experimental_clipping_planes[0]
    
    @property
    def xmaxPlane(self):
        return self.connected_layer.experimental_clipping_planes[1]
    
    @property
    def yminPlane(self):
        return self.connected_layer.experimental_clipping_planes[2]
    
    @property
    def ymaxPlane(self):
        return self.connected_layer.experimental_clipping_planes[3]
    
    def updateX(self):
        xmin, xmax = self.xrange.value()
        self.xminPlane.position = (0,)*(self.connected_layer.ndim-1) + (xmin,)
        self.xmaxPlane.position = (0,)*(self.connected_layer.ndim-1) + (xmax,)
        return None

    def updateY(self):
        ymin, ymax = self.yrange.value()
        self.yminPlane.position = (0,)*(self.connected_layer.ndim-2) + (ymin, 0)
        self.ymaxPlane.position = (0,)*(self.connected_layer.ndim-2) + (ymax, 0)
        return None
