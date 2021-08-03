
import napari
import numpy as np
from qtpy.QtWidgets import QPushButton, QWidget, QGridLayout, QHBoxLayout
import magicgui

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
        @button.clicked.connect
        def _():
            """
            widget = table widget + button widget
            ---------------
            |             |
            |   (table)   | <- table widget
            |             |
            |[Copy][Store]| <- button widget = copy button + store button
            ---------------
            """        
            dfs = list(iter_selected_layer(self.viewer, ["Points", "Tracks"]))
            if len(dfs) == 0:
                return
            for df in dfs:
                widget = QWidget()
                widget.setLayout(QGridLayout())
                axes = list(self.viewer.dims.axis_labels)
                columns = list(df.metadata.get("axes", axes[-df.data.shape[1]:]))
                table = magicgui.widgets.Table(df.data, name=df.name, columns=columns)
                copy_button = QPushButton("Copy")
                copy_button.clicked.connect(lambda: table.to_dataframe().to_clipboard())
                store_button = QPushButton("Store")
                @store_button.clicked.connect
                def _():
                    self.viewer.window.results = table.to_dataframe()
                    return None
                
                button_widget = QWidget()
                layout = QHBoxLayout()
                layout.addWidget(copy_button)
                layout.addWidget(store_button)
                button_widget.setLayout(layout)
                
                widget.layout().addWidget(table.native)
                widget.layout().addWidget(button_widget)
                self.viewer.window.add_dock_widget(widget, area="right", name=df.name)
                
            return None
        
        self.layout().addWidget(button)
        return None
    
    def add_text_button(self):
        button = QPushButton("+Text")
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
        
