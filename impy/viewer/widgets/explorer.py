from __future__ import annotations
from qtpy.QtWidgets import (QWidget, QFileSystemModel, QTreeView, QMenu, QAction, QGridLayout, QLineEdit, QPushButton,
                            QFileDialog)
from qtpy.QtCore import Qt, QModelIndex
import os
import napari
import pyperclip

from .table import read_csv
from ..utils import viewer_imread

# TODO: open txt, open folder

class Explorer(QWidget):
    def __init__(self, viewer:"napari.Viewer", path:str=""):
        super().__init__(viewer.window._qt_window)
        self.viewer = viewer
        self.setLayout(QGridLayout())
        
        self._add_change_root()
        self._add_filetree(path)
        self._add_filter_line()
    
    def _add_change_root(self):
        self.root_button = QPushButton(self)
        self.root_button.setText("Change root directory")
        @self.root_button.clicked.connect
        def _():
            dlg = QFileDialog()
            dlg.setFileMode(QFileDialog.DirectoryOnly)
            dlg.setHistory([self.tree.rootpath])
            dirname = dlg.getExistingDirectory(self, caption="Select root ...", directory=self.tree.rootpath)
            if dirname:
                self.tree.rootpath = dirname
                self.tree._set_file_model(dirname)
        
        self.layout().addWidget(self.root_button)
    
    def _add_filetree(self, path:str):
        path = os.getcwd() if not os.path.exists(path) else path
        self.tree = FileTree(self, path)
        self.layout().addWidget(self.tree)
        return None
    
    def _add_filter_line(self):
        self.line = QLineEdit(self)        
        
        @self.line.editingFinished.connect
        def _():
            self.tree.set_filter(self.line.text())
            
        self.layout().addWidget(self.line)

class FileTree(QTreeView):
    def __init__(self, parent, path:str):
        super().__init__(parent=parent)
        
        self.viewer = parent.viewer
        self.rootpath = path
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.rightClickContextMenu)
        self.setUniformRowHeights(True)
        self.header().hide()

        # Set QFileSystemModel
        self.file_system = QFileSystemModel(self)
        self.file_system.setReadOnly(True)
        self.file_system.setNameFilterDisables(False)            
        self.setExpandsOnDoubleClick(True)
        self._set_file_model(path)
        
        # hide columns except for name
        for i in range(1, self.file_system.columnCount()):
            self.hideColumn(i)
            
        self.clicked.connect(self.onClicked)
        self.doubleClicked.connect(self.onDoubleClicked)
        
        self.show()
    
    def _set_file_model(self, path):
        self.file_system.setRootPath(path)
        self.setModel(self.file_system)
        self.setRootIndex(self.file_system.index(path))
        return None
    
    def rightClickContextMenu(self, point):
        menu = QMenu(self)
        
        open_file = QAction("Open", self)
        open_file.setShortcut(Qt.Key_Return)
        @open_file.triggered.connect
        def _():
            index = self.indexAt(point)
            self.open_path_at(index)
            
        copy_path = QAction("Copy path", self)
        copy_path.setShortcut("Ctrl+C")
        @copy_path.triggered.connect
        def _():
            index = self.indexAt(point)
            self.copy_path_at(index)
        
        menu.addAction(open_file)
        menu.addAction(copy_path)
        menu.exec_(self.mapToGlobal(point))
    
    def open_path_at(self, index:QModelIndex):
        path = self.file_system.filePath(index)
        _, ext = os.path.splitext(path)
        if ext in (".tif", ".tiff", ".mrc", ".rec", ".png", ".jpg"):
            viewer_imread(path)
        elif ext in (".csv", ".dat"):
            read_csv(self.viewer, path)
        return None
        
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
            self.open_path_at(index)
        else:
            if self.isExpanded(index):
                self.collapse(index)
            else:
                self.expand(index)
            return None
    
    def set_filter(self, names:str|list[str]):
        if isinstance(names, str):
            names = names.split(",")
            names = [s.strip() for s in names]
        self.file_system.setNameFilters(names)
        return None
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Return:
            index = self.selected
            index is None or self.onDoubleClicked(index)
        elif event.key() == Qt.Key_C and event.modifiers() == Qt.ControlModifier:
            index = self.selected
            index is None or self.copy_path_at(index)
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
