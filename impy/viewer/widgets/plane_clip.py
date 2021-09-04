from __future__ import annotations
from superqt import QRangeSlider
from qtpy.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QWidget, QFrame, QLabel
from qtpy.QtCore import Qt
import napari

# TODO: reset button

class PlaneClipRange(QWidget):
    def __init__(self, viewer:"napari.Viewer"):
        super().__init__(viewer.window._qt_window)
        self.xrange = QRangeSlider(Qt.Horizontal, parent=self)
        self.yrange = QRangeSlider(Qt.Horizontal, parent=self)
        self.zrange = QRangeSlider(Qt.Horizontal, parent=self)
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
        
        zframe = QFrame(self)
        zframe.setLayout(QHBoxLayout())
        
        zlabel = QLabel(zframe)
        zlabel.setText("z")
        zframe.layout().addWidget(zlabel)
        zframe.layout().addWidget(self.zrange)
        
        self.layout().addWidget(xframe)
        self.layout().addWidget(yframe)
        self.layout().addWidget(zframe)
        
    
    def connectLayer(self, layer:"napari.components.Layer"):
        xmin = layer.extent.data[0,-1]
        xmax = layer.extent.data[1,-1]
        ymin = layer.extent.data[0,-2]
        ymax = layer.extent.data[1,-2]
        zmin = layer.extent.data[0,-3]
        zmax = layer.extent.data[1,-3]
        
        self.xrange.setRange(xmin, xmax)
        self.yrange.setRange(ymin, ymax)
        self.zrange.setRange(zmin, zmax)
        self.xrange.setValue((xmin, xmax))
        self.yrange.setValue((ymin, ymax))
        self.zrange.setValue((zmin, zmax))
        self.connected_layer = layer
        ndim = layer.ndim
        
        self.connected_layer.experimental_clipping_planes = [
            {"position": (0,)*(ndim-1)+(xmin,), "normal": (0, 0, 1), "enabled": True},
            {"position": (0,)*(ndim-1)+(xmax,), "normal": (0, 0, -1), "enabled": True},
            {"position": (0,)*(ndim-2)+(ymin, 0), "normal": (0, 1, 0), "enabled": True},
            {"position": (0,)*(ndim-2)+(ymax, 0), "normal": (0, -1, 0), "enabled": True},
            {"position": (0,)*(ndim-3)+(zmin, 0, 0), "normal": (1, 0, 0), "enabled": True},
            {"position": (0,)*(ndim-3)+(zmax, 0, 0), "normal": (-1, 0, 0), "enabled": True},
            ]
        
        self.xrange.valueChanged.connect(self.updateX)
        self.yrange.valueChanged.connect(self.updateY)
        self.zrange.valueChanged.connect(self.updateZ)
    
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
    
    @property
    def zminPlane(self):
        return self.connected_layer.experimental_clipping_planes[4]
    
    @property
    def zmaxPlane(self):
        return self.connected_layer.experimental_clipping_planes[5]
    
    
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

    def updateZ(self):
        zmin, zmax = self.zrange.value()
        self.zminPlane.position = (0,)*(self.connected_layer.ndim-3) + (zmin, 0, 0)
        self.zmaxPlane.position = (0,)*(self.connected_layer.ndim-3) + (zmax, 0, 0)
        return None