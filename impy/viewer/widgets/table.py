from __future__ import annotations
from typing import Any
import warnings
from qtpy.QtWidgets import (QPushButton, QGridLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QLabel, QCheckBox,
                            QMainWindow, QAction, QHeaderView, QTableWidget, QTableWidgetItem, QStyledItemDelegate,
                            QLineEdit, QSpinBox, QFileDialog, QAbstractItemView)
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

Editable = Qt.ItemFlags(63)    # selectable, editable, drag-enabled, drop-enabled, checkable
NotEditable = Qt.ItemFlags(61) # selectable, not-editable, drag-enabled, drop-enabled, checkable

# TODO: 
# - block 1,2,3,..., [, ] when table is not editable
#   https://stackoverflow.com/questions/48299384/disable-keyevent-for-unneeded-qwidget

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
        self.filter_widget = None
        self._linked_layer = None
        self.plot_settings = dict(x=None, kind="line", legend=True, subplots=False, sharex=False, sharey=False,
                                  logx=False, logy=False, bins=10)
        self.last_plot = "plot"
        self.flag = NotEditable
        
        if df is None or df == {}:
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
        
        self.max_index = df.shape[0]
        
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
        @self.table_native.itemChanged.connect
        def _(item):
            i = item.column()
            self.table_native.resizeColumnToContents(i)
        
        # When horizontal header is double-clicked, table enters edit mode.
        header = self.header
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setSectionsClickable(True)
        header.sectionDoubleClicked.connect(self._edit_header)
        
        # When vertical header is double-clicked, move camera/step in viewer.
        self.table_native.verticalHeader().sectionDoubleClicked.connect(self._linked_callback)
        
        super().__init__(viewer.window._qt_window)
        
        self.menu_bar = self.menuBar()
                
        self._add_table_menu()
        self._add_edit_menu()
        self._add_plot_menu()
        
        self.setCentralWidget(self.table_native)
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setWindowTitle(self.name)
        
        self.setEditability(self.flag)
        
    def __repr__(self):
        return f"TableWidget with data:\n{self.table.to_dataframe().__repr__()}"
    
    @property
    def linked_layer(self) -> "napari.components.Layer":
        return self._linked_layer
    
    @linked_layer.setter
    def linked_layer(self, layer):
        if self.linked_layer is not None:
            raise AttributeError("Cannot set linked layer again.")
        elif not isinstance(layer, (napari.layers.Shapes, napari.layers.Points)):
            raise TypeError(f"Cannot set {type(layer)}")
        
        self._linked_layer = layer
        
        # highlight row(s) if object(s) are selected in the viewer.
        @layer.mouse_drag_callbacks.append
        def link_selection_to_table(_layer, event):
            while event.type != "mouse_release":
                yield
            self._read_selected_data_from_layer(_layer)
        
        # delete row(s) if object(s) are deleted in the viewer.
        def delete_selected_points(layer):
            selected = sorted(layer.selected_data)
            for i in reversed(selected):
                self.table_native.removeRow(i)
            layer.remove_selected()
        
        layer.bind_key("Delete", delete_selected_points, overwrite=True)
        layer.bind_key("Backspace", delete_selected_points, overwrite=True)
        layer.bind_key("1", delete_selected_points, overwrite=True)
        
        # add an empty row if new object is added in the viewer.
        @layer.events.data.connect
        def _(event):
            nrow = self.table_native.rowCount()
            ncol = self.table_native.columnCount()
            if len(event.value) > min(nrow, nrow*ncol):
                self._appendRow()
            self._read_selected_data_from_layer(event.source)
        
        self._connect_item_with_properties()
        layer.metadata.update({"linked_table": self})
        return None        
    
    def _read_selected_data_from_layer(self, layer):
        first_item = None
        self.table_native.clearSelection()
        for i in layer.selected_data:
            self.table_native.selectRow(i)
            if first_item is None:
                first_item = self.table_native.item(i,0)
        
        self.table_native.scrollToItem(first_item, hint=QAbstractItemView.PositionAtTop)
        return None

    @property
    def header_as_tuple(self) -> tuple:
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
        
        df.index.name = "Index"
        return df
    
    def set_header(self, i:int, name:Any):
        # This method is called when
        # - a new column is created
        # - header to row
        newname = str(name)
        self.table_native.setHorizontalHeaderItem(i, QTableWidgetItem(newname))
        return None
    
    def set_header_and_properties(self, i:int, name:Any):
        newname = str(name)
        oldname = str(self.header_as_tuple[i])
        self.table_native.setHorizontalHeaderItem(i, QTableWidgetItem(newname))
        if self.linked_layer is not None:
            prop:dict = self.linked_layer.properties
            prop[newname] = prop.pop(oldname)
            self.linked_layer.properties = prop
        return None
    
    def store_as_dataframe(self, selected=False):
        """
        Send table contents to Results widget.

        Parameters
        ----------
        selected : bool, default is False
            If True, only selected range will be send to results.
        """        
        df = self.to_dataframe(selected)
        self.viewer.window._results.append(df)
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
    
    def save_as_csv(self):
        dlg = QFileDialog()

        hist = napari.utils.history.get_save_history()
        dlg.setHistory(hist)
        
        last_hist = hist[0]
        filename, _ = dlg.getSaveFileName(
            parent=self.viewer.window.qt_viewer,
            caption="Save table as csv",
            directory=last_hist,
        )

        if filename:
            if not filename.endswith(".csv"):
                filename += ".csv"
            self.to_dataframe(False).to_csv(filename)
            napari.utils.history.update_save_history(filename)

        return None
    

    def add_point(self, data="cursor position", size=None, face_color=None, edge_color=None, properties=None, **kwargs):
        """
        Add point in a layer and append its property to the end of the table. They are linked to each other.
        """
        scale = np.array([r[2] for r in self.viewer.dims.range])
        
        if isinstance(data, str) and data == "cursor position":
            data = np.array(self.viewer.cursor.position) / scale
        else:
            data = np.asarray(data)
            if data.ndim != 1:
                raise ValueError("1-D array required.")
        
        nrow = self.table_native.rowCount()
        nrow = 0 if nrow*self.table_native.columnCount() == 0 else nrow
        if properties is None:
            axes = self.viewer.dims.axis_labels[-data.size:]
            properties = {axes[k]: data[k] for k in range(data.size)}
            
        if self.linked_layer is None:
            if nrow > 0:
                raise ValueError("Table already has data. Cannot make a linked layer.")
            
            self.linked_layer = \
                self.viewer.add_points(data, 
                                       properties={k:np.atleast_1d(v) for k, v in properties.items()}, 
                                       scale=scale[-len(data):],
                                       size=5 if size is None else size, 
                                       face_color=[0, 0, 0, 0] if face_color is None else face_color, 
                                       edge_color=[0, 1, 0, 1] if edge_color is None else edge_color, 
                                       name=f"Points from {self.name}", 
                                       n_dimensional=True,
                                       **kwargs)
                            
        elif isinstance(self.linked_layer, napari.layers.Points):
            with self.linked_layer.events.blocker_all():
                self.linked_layer.add(data)
            if size is not None:
                self.linked_layer.current_size = size
            if face_color is not None:
                self.linked_layer.current_face_color = face_color
            if edge_color is not None:
                self.linked_layer.current_edge_color = edge_color
            if properties is not None:
                self.linked_layer.current_properties.update(properties)
        else:
            type_ = self.linked_layer.__class__.__name__.split(".")[-1]
            raise TypeError(f"Table is linked to {type_} layer now. Cannot add points.")
        
        self._appendRow(data=properties)
        return None
    
    def add_shape(self, data, shape_type="rectangle", face_color=None, edge_color=None, 
                  properties=None, **kwargs):
        """
        Add point in a layer and append its property to the end of the table. They are linked to each other.
        """
        nrow = self.table_native.rowCount()
        nrow = 0 if nrow*self.table_native.columnCount() == 0 else nrow
        scale = np.array([r[2] for r in self.viewer.dims.range])
        data = np.asarray(data)
        properties = {"ID": self.max_index} if properties is None else properties
        
        if self.linked_layer is None:
            if nrow > 0:
                raise ValueError("Table already has data. Cannot make a linked layer.")
            
            self.linked_layer = \
                self.viewer.add_shapes([data], 
                                        shape_type=shape_type,
                                        properties={k:np.atleast_1d(v) for k, v in properties.items()}, 
                                        scale=scale[-data.shape[1]:],
                                        face_color=[0, 0, 0, 0] if face_color is None else face_color, 
                                        edge_color=[0, 1, 0, 1] if edge_color is None else edge_color, 
                                        name=f"Shapes from {self.name}",
                                        **kwargs)
            
            @self.linked_layer.events.data.connect
            def _(*args):
                pass
        elif isinstance(self.linked_layer, napari.layers.Shapes):
            with self.linked_layer.events.blocker_all():
                self.linked_layer.add(data, shape_type=shape_type, edge_color=edge_color, face_color=face_color)
            if properties is not None:
                self.linked_layer.current_properties.update(properties)
        else:
            type_ = self.linked_layer.__class__.__name__.split(".")[-1]
            raise TypeError(f"Table is linked to {type_} layer now. Cannot add points.")
        
        self._appendRow(data=properties)
        return None
        
    
    def _linked_callback(self, index:int):
        if self.linked_layer is None:
            return None

        data = np.atleast_2d(self.linked_layer.data[index])
        # update camera    
        scale = self.linked_layer.scale
        center = np.mean(data, axis=0) * scale
        self.viewer.dims.current_step = list(data[0,:].astype(np.int64))

        self.viewer.camera.center = center
        zoom = self.viewer.camera.zoom
        self.viewer.camera.events.zoom() # Here events are emitted and zoom changes automatically.
        self.viewer.camera.zoom = zoom
        
        self.linked_layer.selected_data = {index}
        self.linked_layer._set_highlight()
        
        return None

    def _connect_item_with_properties(self):
        @self.table_native.itemChanged.connect
        def _(item:QTableWidgetItem):
            if not item.isSelected():
                return None
            row = item.row()
            col = item.column()
            colname = str(self.header_as_tuple[col])
            self.linked_layer.properties[colname][row] = item.text()
            
        return None
    
    def plot(self):
        from .._plt import mpl
        from napari.utils.notifications import Notification, notification_manager
        backend = mpl.get_backend()
        mpl.use("Agg")
        try:
            self._add_figuire()
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
        except Exception as e:
            notification_manager.dispatch(Notification.from_exception(e))
        finally:
            mpl.use(backend)
        self.last_plot = "plot"
        return None
    
    def hist(self):
        from .._plt import mpl
        from napari.utils.notifications import Notification, notification_manager
        backend = mpl.get_backend()
        mpl.use("Agg")
        try:
            self._add_figuire()
            
            df = self._get_selected_dataframe()
            
            with warnings.catch_warnings(), mpl.style.context("night"):
                warnings.simplefilter("ignore", UserWarning)
                kw = {k:self.plot_settings[k] for k in ["sharex", "sharey", "bins", "legend"]}
                df.hist(ax=self.ax, grid=True, **kw)
                
                self.fig.tight_layout()
                self.fig.canvas.draw()
                self.figure_widget.show()
        except Exception as e:
            notification_manager.dispatch(Notification.from_exception(e))
        finally:
            mpl.use(backend)
        self.last_plot = "hist"
        return None
    
    
    def restore_linked_layer(self):
        """
        Add linked layer to the viewer again, if it has deleted from the layer list.
        """ 
        if self.linked_layer is None:
            raise ValueError("No linked layer in this table.")
        elif self.linked_layer in self.viewer.layers:
            self.viewer.layers.selection = {self.linked_layer}
        else:
            self.viewer.add_layer(self.linked_layer)
    
    def header_to_row(self):
        """
        Convert table header to the top row.
        """        
        if self.linked_layer is not None:
            raise ValueError("Cannot convert header to row when linked layer exists.")
        self.table_native.insertRow(0)
        for i, item in enumerate(self.header_as_tuple):
            self.table_native.setItem(0, i, QTableWidgetItem(str(item)))
            self.set_header(i, i)
        for i in range(self.table_native.rowCount()):
            self.table_native.setVerticalHeaderItem(i, QTableWidgetItem(str(i)))
        return None
    
    def delete_selected_rows(self):
        """
        Delete all the rows that contain selected cells.
        """        
        rows, cols = self._get_selected()
        for i in reversed(rows):
            self.table_native.removeRow(i)
            
        # BUG: when multiple rows are deleted, wrong points/shapes are deleted
        # If a points layer is linked, also delete points. 
        if self.linked_layer is not None:
            self.linked_layer.selected_data = set(rows)
            self.linked_layer.remove_selected()
        return None

    def delete_selected_columns(self):
        """
        Delete all the columns that contain selected cells.
        """        
        rows, cols = self._get_selected()
        for i in reversed(cols):
            if self.linked_layer is not None:
                colname = str(self.header_as_tuple[i])
                self.linked_layer.properties.pop(colname)
            self.table_native.removeColumn(i)
        return None
        
    def change_plot_setting(self):
        dlg = PlotSetting(self)
        dlg.exec_()
        return None
    
    def appendRow(self, data=None):
        """
        Append a row on the bottom side.
        """        
        if self.linked_layer is not None:
            raise ValueError("Table has a linked layer. Use 'add_point' instead")

        return self._appendRow(data=data)
    
    append = appendRow # for compatibility with other widgets

    def _appendRow(self, data=None):
        """
        Append a row on the bottom side.
        """        

        nrow = self.table_native.rowCount()
        ncol = self.table_native.columnCount()
        
        if ncol == 0:
            return self.newRow(data)
        
        self.table_native.insertRow(nrow)
        self.table_native.setVerticalHeaderItem(nrow, QTableWidgetItem(str(self.max_index)))
        
        self.max_index += 1
        if not hasattr(data, "__len__"):
            data = [""] * ncol
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
            item = QTableWidgetItem(str(item))
            self.table_native.setItem(nrow, i, item)
            item.setFlags(self.flag)
        
        return None
    
    def appendColumn(self, data=None):
        """
        Append a column on the right side. Also can be used to add 1x1 item to an empty table.
        """        
        ncol = self.table_native.columnCount()
        self.table_native.insertColumn(ncol)
        
        # search for an unique name
        colname = ncol
        columns = self.header_as_tuple
        while colname in columns:
            colname += 1
        colname = str(colname)
        
        if not hasattr(data, "__len__"):
            data = [""]*self.table_native.rowCount()
        elif isinstance(data, dict):
            raise TypeError("dict input is not been implemented yet.")
        elif len(data) > self.table_native.rowCount():
            raise ValueError("Input data is longer than the row size.")
        
        for i, item in enumerate(data):
            item = QTableWidgetItem(str(item))
            self.table_native.setItem(i, ncol, item)
            item.setFlags(self.flag)
            
        if self.linked_layer is not None:
            prop:dict = self.linked_layer.properties
            prop[colname] = np.array(data, dtype="<U32")
            self.linked_layer.properties = prop
                
        self.set_header(ncol, colname)
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
            item = QTableWidgetItem(str(data[i]))
            self.table_native.setItem(0, i, item)
            item.setFlags(self.flag)
        
        self.table_native.resizeColumnsToContents()
        return None
    
    def setEditability(self, flag:Qt.ItemFlags):
        self.flag = flag
        self.table_native.clearSelection()
        for i in range(self.table_native.rowCount()):
            for j in range(self.table_native.columnCount()):
                item = self.table_native.item(i, j)
                item.setFlags(flag)
        
        return None
    
    def add_filter(self):
        if self.filter_widget is not None:
            self.filter_widget.show()
            return None
        from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
        
        filter_central = QWidget(self.filter_widget)
        filter_central.setLayout(QHBoxLayout())
        
        filter_label = QLabel(filter_central)
        filter_label.setText("Filter:")
        filter_central.layout().addWidget(filter_label)
        
        self.filter_line = QLineEdit(filter_central)
        self.filter_line.editingFinished.connect(self._run_filter)
        filter_central.layout().addWidget(self.filter_line)
        
        self.filter_widget = QtViewerDockWidget(self, filter_central, name="Table Filter", 
                                                area="top", allowed_areas=["top", "bottom"])
        
        self.addDockWidget(self.filter_widget.qt_area, self.filter_widget)
        return None
    
    def filterRows(self, column_index:int, value:str):
        nrow = self.table_native.rowCount()
        hide = [self.table_native.item(i, column_index).text() != value
                for i in range(nrow)]
        if all(hide):
            for i in range(nrow):
                self.table_native.setRowHidden(i, False)
        else:
            for i in range(nrow):
                self.table_native.setRowHidden(i, hide[i])
        return None
    
    def _run_filter(self):
        _, selected = self._get_selected()
                
        if len(selected) != 1:
            return None
        icol = int(selected[0])
        value = self.filter_line.text()
        self.filterRows(icol, value)
        return None

    def _change_editability(self):
        if self.flag is Editable:
            flag = NotEditable
        else:
            flag = Editable
        return self.setEditability(flag)
        
    def _get_selected_dataframe(self) -> pd.DataFrame:
        """
        Convert selected cells into pandas.DataFrame if possible.
        """        
        sl = self._get_selected()
        try:
            df = self.table.to_dataframe().iloc[sl]
        except TypeError:
            raise ValueError("Table range is not correctly selected")
        return df
    
    def _get_selected(self) -> tuple[list[int], list[int]]:
        """
        Get fancy indexing slices of the selected region.
        """        
        selected:list = self.table_native.selectedRanges() # list of QTableWidgetSelectionRange
        if len(selected) == 0:
            return [], []
        sl_row = set()
        sl_column = set()
        for rng in selected:
            row_range = set(filter(lambda x: (not self.table_native.isRowHidden(x)),
                                   range(rng.topRow(), rng.bottomRow()+1))
                            )
            
            column_range = set(range(rng.leftColumn(), rng.rightColumn()+1))
            sl_row |= row_range
            sl_column |= column_range
        
        return list(sl_row), list(sl_column)
        
    
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
        
        store = QAction("Store selected", self)
        store.triggered.connect(lambda: self.store_as_dataframe(selected=True))

        save = QAction("Save as csv", self)
        save.triggered.connect(self.save_as_csv)
        
        resize = QAction("Resize columns", self)
        resize.triggered.connect(self.table_native.resizeColumnsToContents)
        
        filt = QAction("Filter", self)
        filt.triggered.connect(self.add_filter)
        
        restore = QAction("Restore linked layer", self)
        restore.triggered.connect(self.restore_linked_layer)
                
        close = QAction("Delete widget", self)
        close.triggered.connect(self.delete_self)
        
        self.table_menu.addAction(copy_all)
        self.table_menu.addAction(copy)
        self.table_menu.addAction(store_all)
        self.table_menu.addAction(store)
        self.table_menu.addAction(save)
        self.table_menu.addSeparator()
        self.table_menu.addAction(resize)
        self.table_menu.addAction(restore)
        self.table_menu.addAction(filt)
        self.table_menu.addSeparator()
        self.table_menu.addAction(close)
        return None
    
    
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
        
        change = QAction("Editable", self, checkable=True, checked=False)
        change.triggered.connect(self._change_editability)

        self.edit_menu.addAction(head2row)
        self.edit_menu.addAction(addrow)
        self.edit_menu.addAction(addcol)
        self.edit_menu.addAction(delrow)
        self.edit_menu.addAction(delcol)
        self.edit_menu.addAction(change)
        return None
        
    def _add_plot_menu(self):
        self.plot_menu = self.menu_bar.addMenu("&Plot")
        
        plot = QAction("Plot", self.viewer.window._qt_window)
        plot.triggered.connect(self.plot)
        
        hist = QAction("Histogram", self.viewer.window._qt_window)
        hist.triggered.connect(self.hist)
        
        setting = QAction("Setting ...", self.viewer.window._qt_window)
        setting.triggered.connect(self.change_plot_setting)
        
        self.plot_menu.addAction(plot)
        self.plot_menu.addAction(hist)
        self.plot_menu.addAction(setting)
        return None

    
    def _add_figuire(self):
        from .._plt import EventedCanvas, plt_figure
        
        if self.fig is None:
            from napari._qt.widgets.qt_viewer_dock_widget import QtViewerDockWidget
            
            self.fig = plt_figure()
            canvas = EventedCanvas(self.fig)
            self.figure_widget = QtViewerDockWidget(self, canvas, name="Figure",
                                                    area="bottom", allowed_areas=["right", "bottom"])
            self.figure_widget.setMinimumHeight(120)
            self.addDockWidget(self.figure_widget.qt_area, self.figure_widget)
        else:
            self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        
        return None
    
    def delete_self(self):
        """
        Remove from the dock widget list of the parent viewer.
        """        
        self.removeDockWidget(self.figure_widget)
        if self.linked_layer is not None:
            self.linked_layer.metadata.pop("linked_table", None)
        dock = self.viewer.window._dock_widgets[self.name]
        self.viewer.window.remove_dock_widget(dock)
        return None
    
    def _edit_header(self, i:int):
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
        if self.flag == NotEditable:
            return None
        line = QLineEdit(parent=self.header)

        # set geometry
        edit_geometry = line.geometry()
        edit_geometry.setWidth(self.header.sectionSize(i))
        edit_geometry.moveLeft(self.header.sectionViewportPosition(i))
        line.setGeometry(edit_geometry)
        
        line.setText(str(self.header_as_tuple[i]))
        line.setHidden(False)
        line.setFocus()
        line.selectAll()
        
        self._line = line # we have to retain the pointer, otherwise got error sometimes
        
        @self._line.editingFinished.connect
        def _():
            self._line.setHidden(True)
            self.set_header_and_properties(i, self._line.text())
            self.table_native.resizeColumnToContents(i)
        
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
    

class FilterWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent=parent)
        self.setLayout(QHBoxLayout())
    
    def _add_widgets(self):
        self.line = QLineEdit(self)
        self.line.chan



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
