from __future__ import annotations
import warnings
from qtpy.QtWidgets import (QPushButton, QGridLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QLabel, QCheckBox,
                            QMainWindow, QAction, QHeaderView, QTableWidget, QTableWidgetItem)

import magicgui
import napari
import numpy as np
from functools import partial
import pandas as pd

class TableWidget(QMainWindow):
    """
    +-------------------------------+
    |[Data][Plot]                   |
    |                               |
    |            (table)            |
    +-------------------------------+
    |        (figure canvas)        |
    +-------------------------------+
    """        
    n_table = 0
    def __init__(self, viewer:"napari.viewer.Viewer", df:np.ndarray|pd.DataFrame|dict, columns=None, name=None):
        # TODO: Warning of QMainWindow::count
        self.viewer = viewer
        self.fig = None
        self.ax = None
        self.figure_widget = None
        self.plot_settings = dict(x=None, kind="line", legend=True, subplots=False, sharex=False, sharey=False,
                                  logx=False, logy=False)
        
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        elif isinstance(df, np.ndarray):
            df = np.atleast_2d(df)
        
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
        self.table_native:QTableWidget = self.table.native
        self.table_native.resizeColumnsToContents()
        header = self.table_native.horizontalHeader()
        for i in range(self.table.shape[1]):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
        
        super().__init__(viewer.window._qt_window)
        
        self.menu_bar = self.menuBar()
                
        self._add_data_menu()
        self._add_plot_menu()
        
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle(self.name)
        
        self.setCentralWidget(self.table_native)
    
    def store_as_dataframe(self, selected=False):
        """
        Send table contents to ``self.viewer.window.results``.

        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be send to results.
        """        
        if selected:
            df = self._get_selected_dataframe()
        else:
            df = self.table.to_dataframe()
        self.viewer.window.results = df
        return None
    
    def copy_as_dataframe(self, selected=False):
        """
        Send table contents to clipboard.

        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be send to clipboard.
        """        
        if selected:
            df = self._get_selected_dataframe()
        else:
            df = self.table.to_dataframe()
        df.to_clipboard()
        return None
    
    def plot(self):
        from .._plt import canvas_plot, plt, EventedCanvas, mpl
        backend = mpl.get_backend()
        mpl.use("Agg")
        with canvas_plot():
            if self.fig is None:
                from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
                
                self.fig = plt.figure()
                canvas = EventedCanvas(self.fig)
                self.figure_widget = QtViewerDockWidget(self, canvas, name="Plot",
                                                        area="bottom", allowed_areas=["right", "bottom"])
                
                self.addDockWidget(self.figure_widget.qt_area, self.figure_widget)
            else:
                self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            df = self._get_selected_dataframe()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                if df.shape[1] == 1 and self.plot_settings["x"] == 0:
                    self.plot_settings["x"] = None
                    df.plot(ax=self.ax, grid=True, **self.plot_settings)
                    self.plot_settings["x"] = 0
                else:
                    df.plot(ax=self.ax, grid=True, **self.plot_settings)
                
                self.fig.canvas.draw()
                self.fig.tight_layout()
                self.figure_widget.show()
        
        mpl.use(backend)
        return None
    
    def change_setting(self):
        dlg = PlotSetting(self)
        dlg.exec_()
    
    def appendRow(self, data=None):
        nrow = self.table_native.rowCount()
        self.table_native.insertRow(nrow)
        if not hasattr(data, "__len__"):
            return None
        elif len(data) > self.table_native.columnCount():
            raise ValueError("Input data is longer than the column size.")
        
        for i, item in enumerate(data):
            self.table_native.setItem(nrow, i, QTableWidgetItem(item))
        return None
    
    def appendColumn(self, data=None):
        ncol = self.table_native.columnCount()
        self.table_native.insertColumn(ncol)
        if not hasattr(data, "__len__"):
            return None
        elif len(data) > self.table_native.rowCount():
            raise ValueError("Input data is longer than the row size.")
        
        for i, item in enumerate(data):
            self.table_native.setItem(i, ncol, QTableWidgetItem(item))
        return None
        
    def show(self):
        if self.figure_widget is not None:
            self.figure_widget.show()
        return super().show()
        
    def _get_selected_dataframe(self) -> pd.DataFrame:
        sl = self._get_selected()
        try:
            df = self.table.to_dataframe().iloc[sl]
        except TypeError:
            raise ValueError("Table range is not correctly selected")
        return df
    
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

    
    def _add_data_menu(self):
        self.data_menu = self.menu_bar.addMenu("&Data")
        
        copy_all = QAction("Copy all", self)
        copy_all.triggered.connect(self.copy_as_dataframe)
        copy_all.setShortcut("Ctrl+Shift+C")
        
        copy = QAction("Copy selected", self)
        copy.triggered.connect(partial(self.copy_as_dataframe, selected=True))
        copy.setShortcut("Ctrl+C")
        
        store_all = QAction("Store all", self)
        store_all.triggered.connect(self.store_as_dataframe)
        store_all.setShortcut("Ctrl+Shift+S")
        
        store = QAction("Store all", self)
        store.triggered.connect(partial(self.store_as_dataframe, selected=True))
        store.setShortcut("Ctrl+S")
        
        resize = QAction("Resize Columns", self)
        resize.triggered.connect(self.table_native.resizeColumnsToContents)
        resize.setShortcut("R")
        
        addrow = QAction("Append row", self)
        addrow.triggered.connect(self.appendRow)
        addrow.setShortcut("Alt+R")
        
        addcol = QAction("Append Column", self)
        addcol.triggered.connect(self.appendColumn)
        addcol.setShortcut("Alt+C")
        
        self.data_menu.addAction(copy_all)
        self.data_menu.addAction(copy)
        self.data_menu.addAction(store_all)
        self.data_menu.addAction(store)
        self.data_menu.addAction(resize)
        self.data_menu.addAction(addrow)
        self.data_menu.addAction(addcol)
        
        
    def _add_plot_menu(self):
        self.plot_menu = self.menu_bar.addMenu("&Plot")
        
        plot = QAction("Plot", self.viewer.window._qt_window)
        plot.triggered.connect(self.plot)
        plot.setShortcut("P")
        
        setting = QAction("Setting ...", self.viewer.window._qt_window)
        setting.triggered.connect(self.change_setting)
        
        self.plot_menu.addAction(plot)
        self.plot_menu.addAction(setting)
    
        
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
        
        self.legend = QCheckBox(self)
        self.legend.setText("Show legend")
        self.legend.setChecked(self.table.plot_settings["legend"])
        self.layout().addWidget(self.legend)
        
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
        
        ok_button = QPushButton("OK", self)
        ok_button.clicked.connect(self.ok)
        buttons.layout().addWidget(ok_button)
        
        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.apply)
        buttons.layout().addWidget(apply_button)
        
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.close)
        buttons.layout().addWidget(cancel_button)
        
        self.layout().addWidget(buttons)
        
    def ok(self):
        self._change_setting()
        self.close()
        return None
    
    def apply(self):
        self._change_setting()
        self.table.plot()
        return None
    
    def _change_setting(self):
        out = dict()
        out["x"] = 0 if self.usex.isChecked() else None
        out["kind"] = str(self.kind.currentText())
        for attr in ["legend", "subplots", "sharex", "sharey", "logx", "logy"]:
            out[attr] = getattr(self, attr).isChecked()
        self.table.plot_settings = out
        return None
        