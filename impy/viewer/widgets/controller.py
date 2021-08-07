
import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QWidget, QHBoxLayout

from .table import TableWidget
from ..utils import iter_selected_layer
from ..._const import Const, SetConst

class Controller(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        layout = QHBoxLayout()
        self.setLayout(layout)
        self.add_get_button()
        self.add_text_button()
        self.add_label_button()
    
    def add_get_button(self):
        button = QPushButton("(x,y)")
        button.setToolTip("Show coordinates in a table")
        @button.clicked.connect
        def _():
            dfs = list(iter_selected_layer(self.viewer, ["Points", "Tracks"]))
            if len(dfs) == 0:
                return
            axes = list(self.viewer.dims.axis_labels)
            for df in dfs:
                columns = list(df.metadata.get("axes", axes[-df.data.shape[1]:]))
                
                widget = TableWidget(self.viewer, df.data, columns=columns, name=df.name)
                
                self.viewer.window.add_dock_widget(widget, area="right", name=df.name)
                
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
                return None
            selected = selected[0]
            if not isinstance(selected, napari.layers.Image):
                return None
            img = selected.data
            if hasattr(img, "labels"):
                self.viewer.status = "Image layer already has labels."
                return None
            with SetConst("SHOW_PROGRESS", False):
                img.append_label(np.zeros(img.shape, dtype=np.uint8))
                
            layer = self.viewer.add_labels(img.labels.value, opacity=0.3, scale=selected.scale, name=f"[L]{img.name}",
                                    translate=selected.translate)
            layer.mode = "paint"
            return None
    
        self.layout().addWidget(button)
        return None
    