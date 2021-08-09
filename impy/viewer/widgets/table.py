from __future__ import annotations
from pandas.core.algorithms import isin
from qtpy.QtWidgets import QPushButton, QGridLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QLabel, QCheckBox

import magicgui
import napari
import numpy as np
import pandas as pd

class TableWidget(QWidget):
    """
    +-------------------------------+
    |                               |
    |            (table)            |
    |                               |
    |[Copy][Store][Plot][Setting...]|
    +-------------------------------+
    """        
    n_table = 0
    def __init__(self, viewer:"napari.viewer.Viewer", df:np.ndarray|pd.DataFrame|dict, columns=None, name=None):
        self.viewer = viewer
        self.fig = None
        self.ax = None
        self.figure_widget = None
        self.plot_settings = dict(x=None, kind="line", subplots=False, sharex=False, sharey=False,
                                  logx=False, logy=False)
        
        super().__init__(viewer.window._qt_window)
        self.setLayout(QGridLayout())
        
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        
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
        setting_button = QPushButton("Setting...")
        setting_button.clicked.connect(self.change_setting)
        
        button_widget = QWidget()
        layout = QHBoxLayout()
        layout.addWidget(copy_button)
        layout.addWidget(store_button)
        layout.addWidget(plot_button)
        layout.addWidget(setting_button)
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
        import matplotlib.pyplot as plt
        if self.figure_widget is None:      
            from matplotlib.backends.backend_qt5agg import FigureCanvas
            self.fig = plt.figure()
            self.figure_widget = self.viewer.window.add_dock_widget(FigureCanvas(self.fig), 
                                                                    name=f"Plot of {self.name}",
                                                                    area="right",
                                                                    allowed_areas=["right"])
            self.ax = self.fig.add_subplot(111)
        else:
            self.ax.cla()
            
        sl = self._get_selected()
        df:pd.DataFrame = self.table.to_dataframe().iloc[sl]
        
        with plt.style.context("dark_background"):
            if df.shape[1] == 1 and self.plot_settings["x"] == 0:
                self.plot_settings["x"] = None
                df.plot(ax=self.ax, grid=True, legend=False, **self.plot_settings)
                self.plot_settings["x"] = 0
            else:
                df.plot(ax=self.ax, grid=True, legend=False, **self.plot_settings)
            
            self.ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            self.fig.canvas.draw()
            self.fig.tight_layout()
            
        return None
    
    def change_setting(self):
        dlg = PlotSetting(self)
        dlg.exec_()
    
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
        
class PlotSetting(QDialog):
    def __init__(self, table:TableWidget):
        self.table = table
        super().__init__(table.viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self._add_widgets()
    
    def _add_widgets(self):
        label = QLabel(self)
        label.setText("Set the plotting style.")
        self.layout().addWidget(label)
        
        self.usex = QCheckBox(self)
        self.usex.setText("Left-most column as X-axis")
        self.usex.setChecked(self.table.plot_settings["x"] == 0)
        self.layout().addWidget(self.usex)
        
        combo = QWidget(self)
        combo.setLayout(QHBoxLayout())
        
        self.kind = QComboBox(self)
        self.kind.addItems(["line", "bar", "hist", "box", "kde"])
        self.kind.setCurrentText(self.table.plot_settings["kind"])
        combo.layout().addWidget(self.kind)
        
        label = QLabel(self)
        label.setText("The kind of plot")
        combo.layout().addWidget(label)
        self.layout().addWidget(combo)
        
        self.subplots = QCheckBox(self)
        self.subplots.setText("Subplots")
        self.subplots.setChecked(self.table.plot_settings["subplots"])
        self.layout().addWidget(self.subplots)
        
        self.sharex = QCheckBox(self)
        self.sharex.setText("Share X-axis")
        self.sharex.setChecked(self.table.plot_settings["sharex"])
        self.layout().addWidget(self.sharex)
        
        self.sharey = QCheckBox(self)
        self.sharey.setText("Share Y-axis")
        self.sharey.setChecked(self.table.plot_settings["sharey"])
        self.layout().addWidget(self.sharey)
        
        self.logx = QCheckBox(self)
        self.logx.setText("log-X")
        self.logx.setChecked(self.table.plot_settings["logx"])
        self.layout().addWidget(self.logx)
        
        self.logy = QCheckBox(self)
        self.logy.setText("log-Y")
        self.logy.setChecked(self.table.plot_settings["logy"])
        self.layout().addWidget(self.logy)
        
        buttons = QWidget(self)
        buttons.setLayout(QHBoxLayout())
        
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.ok)
        buttons.layout().addWidget(self.ok_button)
        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.close)
        buttons.layout().addWidget(self.cancel_button)
        
        self.layout().addWidget(buttons)
        
    def ok(self):
        out = dict()
        out["x"] = 0 if self.usex.isChecked() else None
        out["kind"] = str(self.kind.currentText())
        for attr in ["subplots", "sharex", "sharey", "logx", "logy"]:
            out[attr] = getattr(self, attr).isChecked()
        self.table.plot_settings = out
        self.close()
        return out