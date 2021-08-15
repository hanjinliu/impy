from __future__ import annotations
from qtpy.QtWidgets import QWidget, QFileSystemModel, QTreeView, QMenu, QAction, QGridLayout, QLineEdit
from qtpy.QtCore import Qt, QModelIndex
import os
import napari
import pyperclip

from .table import read_csv
from ..utils import add_labeledarray
from ...core import imread, lazy_imread
from ..._const import Const

# TODO: change root path, search, open colored json QPlainTextEdit widget for txt

class Explorer(QWidget):
    def __init__(self, viewer:"napari.Viewer", path:str=""):
        super().__init__(viewer.window._qt_window)
        self.viewer = viewer
        self.setLayout(QGridLayout())
        path = os.getcwd() if not os.path.exists(path) else path
        self.explorer = FileExplorer(self, path)
        self.layout().addWidget(self.explorer)
        
        self._add_filter_line()
    
    def _add_filter_line(self):
        self.line = QLineEdit(self)        
        
        @self.line.editingFinished.connect
        def _():
            names = self.line.text().split(",")
            names = [s.strip() for s in names]
            self.explorer.set_filter(names)
            
        self.layout().addWidget(self.line)

class FileExplorer(QTreeView):
    def __init__(self, parent, path:str):
        super().__init__(parent=parent)
        
        self.viewer = parent.viewer
        self.clicked_index = None
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.rightClickContextMenu)
        self.setUniformRowHeights(True)

        # Set QFileSystemModel
        self.file_system = QFileSystemModel(self)
        self.file_system.setReadOnly(True)
        self.file_system.setRootPath(path)
        self.file_system.setNameFilterDisables(False)
        
        self.setModel(self.file_system)
        self.setUniformRowHeights(True)
        self.setRootIndex(self.file_system.index(path))
        self.setExpandsOnDoubleClick(True)
        
        # hide columns except for name
        for i in range(1, self.file_system.columnCount()):
            self.hideColumn(i)
            
        self.clicked.connect(self.onClicked)
        
        self.doubleClicked.connect(self.onDoubleClicked)
        
        self.show()
    
    def rightClickContextMenu(self, point):
        menu = QMenu(self)
        copy_path = QAction("Copy path", self)
        @copy_path.triggered.connect
        def _():
            index = self.indexAt(point)
            self.copy_path_at(index)
            
        menu.addAction(copy_path)
        menu.exec_(self.mapToGlobal(point))
    
    def copy_path_at(self, index:QModelIndex):
        """
        Copy the absolute path of the file at index. Double quotations are included.
        """        
        path = self.file_system.filePath(index)
        pyperclip.copy('"' + path + '"')
        return None
    
    def onClicked(self, index:QModelIndex):
        path = self.file_system.filePath(index)
        return None
        
    def onDoubleClicked(self, index:QModelIndex):
        path = self.file_system.filePath(index)
        if os.path.isfile(path):
            _, ext = os.path.splitext(path)
            if ext in (".tif", ".tiff", ".mrc", ".rec", ".png", ".jpg"):
                size = os.path.getsize(path)/1e9
                if size < Const["MAX_GB"]:
                    img = imread(path)
                else:
                    img = lazy_imread(path)
                add_labeledarray(self.viewer, img)
            elif ext in (".csv", ".dat"):
                read_csv(self.viewer, path)
            return None
        else:
            if self.isExpanded(index):
                self.collapse(index)
            else:
                self.expand(index)
            return None
    
    def set_filter(self, names:list[str]):
        self.file_system.setNameFilters(names)
        return None
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            index = self.selected
            if index is not None:
                self.onDoubleClicked(index)
        elif event.key()==Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            index = self.selected
            if index is not None:
                self.copy_path_at(index)
        else:
            return super().keyPressEvent(event)
    
    @property
    def selected(self):
        inds = self.selectionModel().selectedIndexes()
        if len(inds) > 0:
            index = inds[0]
        else:
            index = None
        return index
