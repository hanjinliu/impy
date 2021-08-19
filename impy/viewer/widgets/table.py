from __future__ import annotations
from typing import Any
import warnings
from qtpy.QtWidgets import (QPushButton, QGridLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QLabel, QCheckBox,
                            QMainWindow, QAction, QHeaderView, QTableWidget, QTableWidgetItem, QStyledItemDelegate,
                            QLineEdit, QSpinBox)
from qtpy.QtCore import Qt
import magicgui
import napari
import os
import numpy as np
import pandas as pd

def read_csv(viewer:"napari.Viewer", path):
    df = pd.read_csv(path)
    name = os.path.splitext(os.path.basename(path))[0]
    table = TableWidget(viewer, df, name=name)
    return viewer.window.add_dock_widget(table, area="right", name=table.name)

class TableWidget(QMainWindow):
    """
    +-------------------------------+
    |[Table][Data][Plot]            |
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
        self.registered = dict()
        self.plot_settings = dict(x=None, kind="line", legend=True, subplots=False, sharex=False, sharey=False,
                                  logx=False, logy=False, bins=10)
        self.last_plot = "plot"
        
        if df is None:
            if columns is None:
                df = np.atleast_2d([])
            else:
                df = np.atleast_2d([""]*len(columns))
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
        
        # When horizontal header is double-clicked, table enters edit mode.
        header = self.header
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionsClickable(True)
        header.sectionDoubleClicked.connect(self.edit_header)
        
        # When vertical header is double-clicked, move camera/step in viewer.
        self.table_native.verticalHeader().sectionDoubleClicked.connect(self._move_in_viewer)
        
        super().__init__(viewer.window._qt_window)
        
        self.menu_bar = self.menuBar()
                
        self._add_table_menu()
        self._add_edit_menu()
        self._add_plot_menu()
        
        self.setCentralWidget(self.table_native)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle(self.name)
        
    
    def __repr__(self):
        return f"TableWidget with data:\n{self.table.to_dataframe().__repr__()}"
    
    @property
    def columns(self) -> tuple:
        return self.table.column_headers
    
    @property
    def header(self) -> QHeaderView:
        return self.table_native.horizontalHeader()
    
    def __array__(self) -> np.ndarray:
        return self.table.to_dataframe().values
    
    def to_dataframe(self, selected=False) -> pd.DataFrame:
        """
        Convert table to ``pandas.DataFrame``.
        
        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be converted.
        """        
        if selected:
            df = self._get_selected_dataframe()
        else:
            df = self.table.to_dataframe()
        return df
    
    def set_header(self, i:int, name:Any):
        self.table_native.setHorizontalHeaderItem(i, QTableWidgetItem(str(name)))
        return None
    
    def store_as_dataframe(self, selected=False):
        """
        Send table contents to ``self.viewer.window.results``.

        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be send to results.
        """        
        self.viewer.window.results = self.to_dataframe(selected)
        return None
    
    def copy_as_dataframe(self, selected=False):
        """
        Send table contents to clipboard.

        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be send to clipboard.
        """        
        self.to_dataframe(selected).to_clipboard()
        return None
    
    def link(self, row_index:int=-1, *, step=None, center=None, zoom=None, angles=None):
        """
        Link table row index and viewer states.

        Parameters
        ----------
        row_index : int, default is -1
            Specify which index of row will be linked to the viewer states.
        step : sequence, optional
            If provided, the provided value will be registered instead of ``viewer.dims.current_step``.
        center : sequence, optional
            If provided, the provided value will be registered instead of ``viewer.camera.center``.
        zoom : sequence, optional
            If provided, the provided value will be registered instead of ``viewer.camera.zoom``.
        angles : sequence, optional
            If provided, the provided value will be registered instead of ``viewer.camera.angles``.

        """        
        ncol = self.table_native.rowCount()
        if row_index < 0:
            row_index = self.table.row_headers[ncol + row_index]
        elif row_index not in self.table.row_headers:
            raise IndexError(f"Index {row_index} does not exist in the table '{self.name}'")
            
        step = self.viewer.dims.current_step if step is None else tuple(step)
        center = self.viewer.camera.center if center is None else tuple(center)
        zoom = self.viewer.camera.zoom if zoom is None else float(zoom)
        angles = self.viewer.camera.angles if angles is None else tuple(angles)
        
        state = dict(step=step, center=center, zoom=zoom, angles=angles)
        self.registered[row_index] = state
        
        return None
    
    def _move_in_viewer(self, index:int):
        rowi = self.table.row_headers[index]
        if rowi in self.registered.keys():
            state = self.registered[rowi]
            self.viewer.dims.current_step = state["step"]
            self.viewer.camera.center = state["center"]
            self.viewer.camera.zoom = state["zoom"]
            self.viewer.camera.angles = state["angles"]
        return None
    
    def plot(self):
        from .._plt import EventedCanvas, mpl, plt_figure
        backend = mpl.get_backend()
        mpl.use("Agg")
        try:
            if self.fig is None:
                from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
                
                self.fig = plt_figure()
                canvas = EventedCanvas(self.fig)
                self.figure_widget = QtViewerDockWidget(self, canvas, name="Figure",
                                                        area="bottom", allowed_areas=["right", "bottom"])
                
                self.addDockWidget(self.figure_widget.qt_area, self.figure_widget)
            else:
                self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            df = self._get_selected_dataframe()
            
            with warnings.catch_warnings(), mpl.style.context("night"):
                warnings.simplefilter("ignore", UserWarning)
                kw = self.plot_settings.copy()
                kw.pop("bins")
                if df.shape[1] == 1 and kw["x"] == 0:
                    kw["x"] = None
                    df.plot(ax=self.ax, grid=True, **kw)
                    kw["x"] = 0
                else:
                    df.plot(ax=self.ax, grid=True, **kw)
                
                self.fig.tight_layout()
                self.fig.canvas.draw()
                self.figure_widget.show()
        except Exception:
            mpl.use(backend)
            raise
        else:
            mpl.use(backend)
        self.last_plot = "plot"
        return None
    
    def hist(self):
        from .._plt import EventedCanvas, mpl, plt_figure
        backend = mpl.get_backend()
        mpl.use("Agg")
        try:
            if self.fig is None:
                from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
                
                self.fig = plt_figure()
                canvas = EventedCanvas(self.fig)
                self.figure_widget = QtViewerDockWidget(self, canvas, name="Figure",
                                                        area="bottom", allowed_areas=["right", "bottom"])
                
                self.addDockWidget(self.figure_widget.qt_area, self.figure_widget)
            else:
                self.fig.clf()
            self.ax = self.fig.add_subplot(111)
            
            df = self._get_selected_dataframe()
            
            with warnings.catch_warnings(), mpl.style.context("night"):
                warnings.simplefilter("ignore", UserWarning)
                kw = {k:self.plot_settings[k] for k in ["sharex", "sharey", "bins", "legend"]}
                df.hist(ax=self.ax, grid=True, **kw)
                
                self.fig.tight_layout()
                self.fig.canvas.draw()
                self.figure_widget.show()
        except Exception:
            mpl.use(backend)
            raise
        else:
            mpl.use(backend)
        self.last_plot = "hist"
        return None
    
    def header_to_row(self):
        self.table_native.insertRow(0)
        for i, item in enumerate(self.columns):
            self.table_native.setItem(0, i, QTableWidgetItem(str(item)))
            self.set_header(i, i)
        for i in range(self.table_native.rowCount()):
            self.table_native.setVerticalHeaderItem(i, QTableWidgetItem(str(i)))
        return None
    
    def delete_selected_rows(self):
        rows, cols = self._get_selected()
        for i in reversed(rows):
            self.table_native.removeRow(i)
        return None

    def delete_selected_columns(self):
        rows, cols = self._get_selected()
        for i in reversed(cols):
            self.table_native.removeColumn(i)
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
    
    append = appendRow # for compatibility
    
    def appendColumn(self, data=None):
        """
        Append a column on the right side. Also can be used to add 1x1 item to an empty table.
        """        
        ncol = self.table_native.columnCount()
        self.table_native.insertColumn(ncol)
        self.set_header(ncol, ncol)
        
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
            self.set_header(i, h)
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
        return list(sl_row), list(sl_column)

    
    def _add_edit_menu(self):
        self.edit_menu = self.menu_bar.addMenu("&Edit")
        
        addrow = QAction("Append row", self)
        addrow.triggered.connect(self.appendRow)
        addrow.setShortcut("Alt+R")
        
        addcol = QAction("Append column", self)
        addcol.triggered.connect(self.appendColumn)
        addcol.setShortcut("Alt+C")
        
        head2row = QAction("Header to top row", self)
        head2row.triggered.connect(self.header_to_row)
        
        delrow = QAction("Delete selected rows", self)
        delrow.triggered.connect(self.delete_selected_rows)
        
        delcol = QAction("Delete selected columns", self)
        delcol.triggered.connect(self.delete_selected_columns)

        self.edit_menu.addAction(head2row)
        self.edit_menu.addAction(addrow)
        self.edit_menu.addAction(addcol)
        self.edit_menu.addAction(delrow)
        self.edit_menu.addAction(delcol)
        
    
    def _add_table_menu(self):
        self.table_menu = self.menu_bar.addMenu("&Table")
        
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
        
        resize = QAction("Resize Columns", self)
        resize.triggered.connect(self.table_native.resizeColumnsToContents)
        resize.setShortcut("R")
                
        close = QAction("Delete widget", self)
        close.triggered.connect(self.delete_self)
        
        self.table_menu.addAction(copy_all)
        self.table_menu.addAction(copy)
        self.table_menu.addAction(store_all)
        self.table_menu.addAction(store)
        self.table_menu.addAction(resize)
        self.table_menu.addAction(close)
        
    def _add_plot_menu(self):
        self.plot_menu = self.menu_bar.addMenu("&Plot")
        
        plot = QAction("Plot", self.viewer.window._qt_window)
        plot.triggered.connect(self.plot)
        plot.setShortcut("P")
        
        hist = QAction("Histogram", self.viewer.window._qt_window)
        hist.triggered.connect(self.hist)
        hist.setShortcut("H")
        
        setting = QAction("Setting ...", self.viewer.window._qt_window)
        setting.triggered.connect(self.change_plot_setting)
        
        self.plot_menu.addAction(plot)
        self.plot_menu.addAction(hist)
        self.plot_menu.addAction(setting)
    
    def delete_self(self):
        self.removeDockWidget(self.figure_widget)
        dock = self.viewer.window._dock_widgets[self.name]
        self.viewer.window.remove_dock_widget(dock)
        return None
    
    def edit_header(self, i:int):
        """
        Enter edit header mode when a header item is double-clicked.

        Parameters
        ----------
        i : int
            The index of header item that is double-clicked.

        References
        ----------
        - https://www.qtcentre.org/threads/42388-Make-QHeaderView-Editable
        """        
        line = QLineEdit(parent=self.header)

        # set geometry
        edit_geometry = line.geometry()
        edit_geometry.setWidth(self.header.sectionSize(i))
        edit_geometry.moveLeft(self.header.sectionViewportPosition(i))
        line.setGeometry(edit_geometry)
        
        line.setText(str(self.columns[i]))
        line.setHidden(False)
        line.setFocus()
        line.selectAll()
        
        self._line = line # we have to retain the pointer, otherwise got error sometimes
        
        @self._line.editingFinished.connect
        def _():
            self._line.setHidden(True)
            self.set_header(i, self._line.text())
        
        return None

class PlotSetting(QDialog):
    def __init__(self, table:TableWidget):
        self.table = table
        super().__init__(table.viewer.window._qt_window)
        self.resize(180, 120)
        self.setLayout(QGridLayout())
        self.add_widgets()
        
    def add_widgets(self):
        label = QLabel(self)
        label.setText("Set the plotting style.")
        self.layout().addWidget(label)
        
        self.usex = self._add_checkbox(text="Left-most column as X-axis",
                                       checked=(self.table.plot_settings["x"] == 0))
        
        combo = QWidget(self)
        combo.setLayout(QHBoxLayout())
        
        self.kind = QComboBox(self)
        self.kind.addItems(["line", "bar", "box", "kde"])
        self.kind.setCurrentText(self.table.plot_settings["kind"])
        combo.layout().addWidget(self.kind)
        
        label = QLabel(self)
        label.setText("The kind of plot")
        combo.layout().addWidget(label)
        self.layout().addWidget(combo)
        
        self.bins = QSpinBox(self)
        self.bins.setRange(2, 100)
        self.bins.setValue(self.table.plot_settings["bins"])
        self.layout().addWidget(self.bins)

        self.legend = self._add_checkbox(text="Show legend",
                                         checked=self.table.plot_settings["legend"])
        self.subplots = self._add_checkbox(text="Subplots",
                                           checked=self.table.plot_settings["subplots"])
        self.sharex = self._add_checkbox(text="Share X-axis",
                                         checked=self.table.plot_settings["sharex"])
        self.sharey = self._add_checkbox(text="Share Y-axis",
                                         checked=self.table.plot_settings["sharey"])
        self.logx = self._add_checkbox(text="log-X",
                                       checked=self.table.plot_settings["logx"])
        self.logy = self._add_checkbox(text="log-Y",
                                       checked=self.table.plot_settings["logy"])
        self._add_buttons()
    
    
    def ok(self):
        self.change_setting()
        self.close()
        return None

    def apply(self):
        self.change_setting()
        getattr(self.table, self.table.last_plot)()
        return None
    
    def change_setting(self):
        out = dict()
        out["x"] = 0 if self.usex.isChecked() else None
        out["kind"] = str(self.kind.currentText())
        out["bins"] = self.bins.value()
        for attr in ["legend", "subplots", "sharex", "sharey", "logx", "logy"]:
            out[attr] = getattr(self, attr).isChecked()
        self.table.plot_settings.update(out)
        return None
    
    def _add_checkbox(self, text:str, checked:bool):
        checkbox = QCheckBox(self)
        checkbox.setText(text)
        checkbox.setChecked(checked)
        self.layout().addWidget(checkbox)
        return checkbox
    
    def _add_buttons(self):
        buttons = QWidget(self)
        buttons.setLayout(QHBoxLayout())
        
        ok_button = QPushButton("Save and close", self)
        ok_button.clicked.connect(self.ok)
        buttons.layout().addWidget(ok_button)
        
        apply_button = QPushButton("Apply", self)
        apply_button.clicked.connect(self.apply)
        buttons.layout().addWidget(apply_button)
        
        cancel_button = QPushButton("Cancel", self)
        cancel_button.clicked.connect(self.close)
        buttons.layout().addWidget(cancel_button)
        
        self.layout().addWidget(buttons)
    
    

class FloatDelegate(QStyledItemDelegate):
    """
    This class is used for displaying table widget items. With this float will be displayed as a
    formated string.
    """    
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
