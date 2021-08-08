from qtpy.QtWidgets import QPushButton, QGridLayout, QHBoxLayout, QWidget
import magicgui
import napari
import pandas as pd

class TableWidget(QWidget):
    """
    widget = table widget + button widget
    ---------------
    |             |
    |   (table)   | <- table widget
    |             |
    |[Copy][Store]| <- button widget = copy button + store button
    ---------------
    """        
    def __init__(self, viewer:"napari.viewer.Viewer", df, columns=None, name=None):
        self.viewer = viewer
        super().__init__(viewer.window._qt_window)
        self.setLayout(QGridLayout())
        
        if columns is None:
            if isinstance(df, pd.DataFrame):
                columns = list(df.columns)
            else:
                columns = list(chr(i) for i in range(65, 65+df.shape[1]))
                
        if name is None:
            name = "Table"
        
        if isinstance(df, pd.DataFrame):
            data = df.values
        else:
            data = df
        self.table = magicgui.widgets.Table(data, name=name, columns=columns)
        self.table_native = self.table.native
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_as_dataframe)
        store_button = QPushButton("Store")
        store_button.clicked.connect(self.store_as_dataframe)
        
        button_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(copy_button)
        layout.addWidget(store_button)
        button_widget.setLayout(layout)
        
        self.layout().addWidget(self.table_native)
        self.layout().addWidget(button_widget)
    
    def store_as_dataframe(self):
        self.viewer.window.results = self.table.to_dataframe()
        return None
    
    def copy_as_dataframe(self):
        self.table.to_dataframe().to_clipboard()
        return None
    
    def _get_selected(self):
        selected = self.table_native.selectedRanges()
        sl = [selected.leftColumns(), selected.rightColumns()]
        ...
        
    
    