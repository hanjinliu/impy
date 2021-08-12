from __future__ import annotations
import warnings
from qtpy.QtWidgets import (QPushButton, QGridLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QLabel, QCheckBox,
                            QMainWindow, QAction, QHeaderView, QTableWidget, QTableWidgetItem, QStyledItemDelegate)
import magicgui
import napari
import numpy as np
import pandas as pd

# TODO: rename column, Warning of QMainWindowLayout::count

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
    def __init__(self, viewer:"napari.Viewer", df:np.ndarray|pd.DataFrame|dict, columns=None, name=None):
        self.viewer = viewer
        self.fig = None
        self.ax = None
        self.figure_widget = None
        self.plot_settings = dict(x=None, kind="line", legend=True, subplots=False, sharex=False, sharey=False,
                                  logx=False, logy=False)
        if df is None:
            df = np.atleast_2d([])
        elif isinstance(df, dict):
            if np.isscalar(next(iter(df.values()))):
                df = pd.DataFrame(df, index=[0])
            else:
                df = pd.DataFrame(df)
        elif not isinstance(df, pd.DataFrame):
            df = np.atleast_2d(df)
        
        if columns is None:
            if isinstance(df, pd.DataFrame):
                columns = list(df.columns)
            else:
                columns = list(range(df.shape[1]))
        
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
        self.table_native.setItemDelegate(FloatDelegate(parent=self.table_native))
        self.table_native.resizeColumnsToContents()
        header = self.table_native.horizontalHeader()
        
        for i in range(self.table.shape[1]):
            header.setSectionResizeMode(i, QHeaderView.Fixed)
        
        super().__init__(viewer.window._qt_window)
        
        self.menu_bar = self.menuBar()
                
        self._add_table_menu()
        self._add_data_menu()
        self._add_plot_menu()
        
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle(self.name)
        self.setCentralWidget(self.table_native)
        
    
    def __repr__(self):
        return f"TableWidget with data:\n{self.table.to_dataframe().__repr__()}"
    
    @property
    def columns(self):
        return self.table.column_headers
    
    @columns.setter
    def columns(self, value):
        self.table.column_headers = value
        return None
        
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
    
    def change_plot_setting(self):
        dlg = PlotSetting(self)
        dlg.exec_()
    
    def appendRow(self, data=None):
        """
        Append a row on the bottom side.
        """        
        nrow = self.table_native.rowCount()
        ncol = self.table_native.columnCount()
        
        if ncol == 0:
            return self.newRow(data)
        
        self.table_native.insertRow(nrow)
        self.table_native.setVerticalHeaderItem(nrow, QTableWidgetItem(str(nrow)))
        
        if not hasattr(data, "__len__"):
            return None
        elif isinstance(data, dict):
            header = self.table.column_headers
            data_ = [""] * len(header)
            for k, v in data.items():
                i = header.index(k)
                data_[i] = v
            data = data_
        elif len(data) > self.table_native.columnCount():
            raise ValueError("Input data is longer than the column size.")
        
        for i, item in enumerate(data):
            self.table_native.setItem(nrow, i, QTableWidgetItem(str(item)))
        return None
    
    append = appendRow
    
    def appendColumn(self, data=None):
        """
        Append a column on the right side. Also can be used to add 1x1 item to an empty table.
        """        
        ncol = self.table_native.columnCount()
        self.table_native.insertColumn(ncol)
        self.table_native.setHorizontalHeaderItem(ncol, QTableWidgetItem(str(ncol)))
        self.table_native.horizontalHeader().setSectionResizeMode(ncol, QHeaderView.Fixed)
        
        if not hasattr(data, "__len__"):
            return None
        elif isinstance(data, dict):
            raise TypeError("dict input is not been implemented yet.")
        elif len(data) > self.table_native.rowCount():
            raise ValueError("Input data is longer than the row size.")
        
        for i, item in enumerate(data):
            self.table_native.setItem(i, ncol, QTableWidgetItem(str(item)))
        
        self.table_native.resizeColumnsToContents()
        
        return None
    
    def newRow(self, data):
        """
        Add a new row to an empty table.
        """        
        if not hasattr(data, "__len__"):
            # add 1x1 empty item
            return self.appendColumn()
        elif isinstance(data, dict):
            header = list(data.keys())
            data = list(data.values())
        else:
            header = np.arange(len(data))
        
        for i, h in enumerate(header):
            self.table_native.insertColumn(i)
            self.table_native.horizontalHeader().setSectionResizeMode(i, QHeaderView.Fixed)
            self.table_native.setHorizontalHeaderItem(i, QTableWidgetItem(str(h)))
            self.table_native.setItem(0, i, QTableWidgetItem(str(data[i])))
        
        self.table_native.resizeColumnsToContents()
        return None
        
        
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
        copy.triggered.connect(lambda: self.copy_as_dataframe(selected=True))
        copy.setShortcut("Ctrl+C")
        
        store_all = QAction("Store all", self)
        store_all.triggered.connect(self.store_as_dataframe)
        store_all.setShortcut("Ctrl+Shift+S")
        
        store = QAction("Store selected", self)
        store.triggered.connect(lambda: self.store_as_dataframe(selected=True))
        store.setShortcut("Ctrl+S")
        
        self.data_menu.addAction(copy_all)
        self.data_menu.addAction(copy)
        self.data_menu.addAction(store_all)
        self.data_menu.addAction(store)
        
    
    def _add_table_menu(self):
        self.table_menu = self.menu_bar.addMenu("&Table")
        
        resize = QAction("Resize Columns", self)
        resize.triggered.connect(self.table_native.resizeColumnsToContents)
        resize.setShortcut("R")
        
        addrow = QAction("Append row", self)
        addrow.triggered.connect(self.appendRow)
        addrow.setShortcut("Alt+R")
        
        addcol = QAction("Append Column", self)
        addcol.triggered.connect(self.appendColumn)
        addcol.setShortcut("Alt+C")
        
        close = QAction("Delete Widget", self)
        close.triggered.connect(self.delete_self)
            
        self.table_menu.addAction(resize)
        self.table_menu.addAction(addrow)
        self.table_menu.addAction(addcol)
        self.table_menu.addAction(close)
        
    def _add_plot_menu(self):
        self.plot_menu = self.menu_bar.addMenu("&Plot")
        
        plot = QAction("Plot", self.viewer.window._qt_window)
        plot.triggered.connect(self.plot)
        plot.setShortcut("P")
        
        setting = QAction("Setting ...", self.viewer.window._qt_window)
        setting.triggered.connect(self.change_plot_setting)
        
        self.plot_menu.addAction(plot)
        self.plot_menu.addAction(setting)
    
    def delete_self(self):
        self.removeDockWidget(self.figure_widget)
        dock = self.viewer.window._dock_widgets[self.name]
        self.viewer.window.remove_dock_widget(dock)
    
        
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

class FloatDelegate(QStyledItemDelegate):
    def __init__(self, ndigit=3, parent=None):
        super().__init__(parent=parent)
        self.ndigit = ndigit

    def displayText(self, value, locale):
        value = _convert_type(value)
        if isinstance(value, (int, float)):
            if 0.1 <= abs(value) < 10000 or value == 0:
                if isinstance(value, int):
                    value = str(value)
                else:
                    value = float(value)
                    value = f"{value:.{self.ndigit}f}"
            else:
                value = f"{value:.{self.ndigit}e}"
        
        return super().displayText(value, locale)
    
def _convert_type(value:str):
    if value is None:
        return None
    try:
        out = int(value)
    except ValueError:
        try:
            out = float(value)
        except ValueError:
            out = value
    
    return out