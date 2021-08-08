from __future__ import annotations
from qtpy.QtWidgets import QPushButton, QGridLayout, QHBoxLayout, QWidget
import magicgui
import napari
import pandas as pd

class TableWidget(QWidget):
    n_table = 0
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
        self.fig = None
        self.ax = None
        self.figure_widget = None
        super().__init__(viewer.window._qt_window)
        self.setLayout(QGridLayout())
        
        if columns is None:
            if isinstance(df, pd.DataFrame):
                columns = list(df.columns)
            else:
                columns = list(chr(i) for i in range(65, 65+df.shape[1]))
        
        if name is None:
            self.name = f"Table-{self.__class__.n_table}"
            self.__class__.n_table += 1
        else:
            self.name = name
        
        if isinstance(df, pd.DataFrame):
            data = df.values
        else:
            data = df
        self.table = magicgui.widgets.Table(data, name=self.name, columns=columns)
        self.table_native = self.table.native
        copy_button = QPushButton("Copy")
        copy_button.clicked.connect(self.copy_as_dataframe)
        store_button = QPushButton("Store")
        store_button.clicked.connect(self.store_as_dataframe)
        plot_button = QPushButton("Plot")
        plot_button.clicked.connect(self.plot)
        
        button_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(copy_button)
        layout.addWidget(store_button)
        layout.addWidget(plot_button)
        button_widget.setLayout(layout)
        
        self.layout().addWidget(self.table_native)
        self.layout().addWidget(button_widget)
    
    def store_as_dataframe(self):
        self.viewer.window.results = self.table.to_dataframe()
        return None
    
    def copy_as_dataframe(self):
        self.table.to_dataframe().to_clipboard()
        return None
    
    def plot(self):
        if self.figure_widget is None:
            import matplotlib.pyplot as plt        
            from matplotlib.backends.backend_qt5agg import FigureCanvas
            self.fig = plt.figure()
            self.figure_widget = self.viewer.window.add_dock_widget(FigureCanvas(self.fig), 
                                                                    name=f"Plot of {self.name}",
                                                                    area="right")
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax.cla()
            
        sl = self._get_selected()
        df:pd.DataFrame = self.table.to_dataframe().iloc[sl]
        df.plot(ax=self.ax)
        self.fig.canvas.draw()
        self.fig.canvas.flush_event()
        return None
    
    def _get_selected(self) -> tuple[list[int], list[int]]:
        selected:list = self.table_native.selectedRanges() # list of QTableWidgetSelectionRange
        if len(selected) == 0:
            return None
        sl_row = set()
        sl_column = set()
        for rng in selected:
            row_range = set(range(rng.topRow(), rng.bottomRow()+1))
            column_range = set(range(rng.leftColumn(), rng.rightColumn()+1))
            sl_row |= row_range
            sl_column |= column_range
        
        n_selected = len(self.table_native.selectedItems())
        if len(sl_row) * len(sl_column) != n_selected:
            raise ValueError("Wrong selection range.")
        return list(sl_row), list(sl_column)
        
        
    
    