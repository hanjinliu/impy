
import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QWidget, QHBoxLayout

from .table import TableWidget
from ..utils import iter_selected_layer
from ..._const import Const, SetConst

class Controller(QWidget):
    def __init__(self, viewer:"napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.add_get_coords_button()
        self.add_get_props_button()
        self.add_text_button()
        self.add_label_button()
        self.add_table_button()
    
    def add_get_coords_button(self):
        button = QPushButton("(x,y)")
        button.setToolTip("Show coordinates in a table")
        @button.clicked.connect
        def _():
            layers = list(iter_selected_layer(self.viewer, ["Points", "Tracks"]))
            if len(layers) == 0:
                raise ValueError("No Points or Tracks layer selected")
            axes = list(self.viewer.dims.axis_labels)
            for layer in layers:
                columns = list(layer.metadata.get("axes", axes[-layer.data.shape[1]:]))
                
                widget = TableWidget(self.viewer, layer.data, columns=columns, name=layer.name)
                
                self.viewer.window.add_dock_widget(widget, area="right", name=layer.name)
                
            return None
        
        self.layout().addWidget(button)
        return None
    
    def add_get_props_button(self):
        button = QPushButton("Props")
        button.setToolTip("Show properties in a table")
        @button.clicked.connect
        def _():
            layers = list(iter_selected_layer(self.viewer, ["Points", "Tracks", "Shapes", "Labels"]))
            if len(layers) == 0:
                raise ValueError("No Points, Tracks or Shapes layer selected")
            
            for layer in layers:
                widget = TableWidget(self.viewer, layer.properties, name=layer.name)
                self.viewer.window.add_dock_widget(widget, area="right", name=layer.name)
                
            return None
        
        self.layout().addWidget(button)
        return None
    
    def add_text_button(self):
        button = QPushButton("+Text")
        button.setToolTip("Add a text layer")
        @button.clicked.connect
        def _():
            layer = self.viewer.add_shapes(ndim=2, shape_type="rectangle", name="Text Layer")
            layer.mode = "add_rectangle"
            layer.blending = "additive"
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
        
        self.layout().addWidget(button)
        return None
    
    def add_label_button(self):
        button = QPushButton("Label")
        button.setToolTip("Manually label ImgArray on the viewer")
        @button.clicked.connect
        def _():
            selected = list(self.viewer.layers.selection)
            if len(selected) != 1:
                raise ValueError("No layer selected")
            selected = selected[0]
            if not isinstance(selected, napari.layers.Image):
                raise TypeError("Selected layer is not an image layer")
            img = selected.data
            if hasattr(img, "labels"):
                raise ValueError("Image layer already has labels.")
            with SetConst("SHOW_PROGRESS", False):
                img.append_label(np.zeros(img.shape, dtype=np.uint8))
                
            layer = self.viewer.add_labels(img.labels.value, opacity=0.3, scale=selected.scale, 
                                           name=f"[L]{img.name}", translate=selected.translate,
                                           metadata={"destination_image": img})
            layer.mode = "paint"
            return None
    
        self.layout().addWidget(button)
        return None
    
    def add_table_button(self):
        button = QPushButton("Table")
        button.setToolTip("Add empty table")
        @button.clicked.connect
        def _():
            from .table import TableWidget
            table = TableWidget(self.viewer, None)
            self.viewer.window.add_dock_widget(table, area="right", name=table.name)
            return None
    
        self.layout().addWidget(button)
        return None