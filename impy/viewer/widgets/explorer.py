from __future__ import annotations
from qtpy.QtWidgets import (QWidget, QFileSystemModel, QTreeView, QMenu, QAction, QGridLayout, QLineEdit, QPushButton,
                            QFileDialog, QHBoxLayout, QLabel)
from qtpy.QtCore import Qt, QModelIndex
import os
import napari
try:
    import pyperclip
except ImportError:
    pass

from .table import read_csv
from .textedit import read_txt
from ..utils import viewer_imread, add_labeledarray
from ...core import imread

class Explorer(QWidget):
    """
    A Read-only explorer widget. Capable of filter, set working directory, copy path and open file in the viewer.
    By default QTreeView supports real time update on file change.
    """
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
                napari.utils.history.update_open_history(dirname)
            return None
        
        self.layout().addWidget(self.root_button)
        return None
    
    def _add_filetree(self, path:str=""):
        """
        Add tree view of files with root directory set to ``path``.

        Parameters
        ----------
        path : str, default is ""
            Path of the root directory. If not found, current directory will be used.
        """        
        path = os.getcwd() if not os.path.exists(path) else path
        self.tree = FileTree(self, path)
        self.layout().addWidget(self.tree)
        return None
    
    def _add_filter_line(self):
        """
        Add line edit widget which filters file tree by file names.
        """        
        wid = QWidget(self)
        wid.setLayout(QHBoxLayout())
        
        # add label
        label = QLabel(self)
        label.setText("Filter file name: ")
        
        self.line = QLineEdit(self)
        self.line.setToolTip("Filter by names split by comma. e.g. '*.tif, *csv'.")
        
        @self.line.editingFinished.connect
        def _():
            self.tree.set_filter(self.line.text())
        
        
        wid.layout().addWidget(label)
        wid.layout().addWidget(self.line)
        
        self.layout().addWidget(wid)

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
        self._set_file_model(path)
        
        # hide columns except for name
        for i in range(1, self.file_system.columnCount()):
            self.hideColumn(i)
            
        self.clicked.connect(self.onClicked)
        self.doubleClicked.connect(self.onDoubleClicked)
        
        self.show()
    
    def _set_file_model(self, path:str):
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
        return None
    
    def open_path_at(self, index:QModelIndex):
        path = self.file_system.filePath(index)
        if os.path.isdir(path):
            img = imread(os.path.join(path, "*.tif"))
            add_labeledarray(self.viewer, img)
        _, ext = os.path.splitext(path)
        if ext in (".tif", ".tiff", ".mrc", ".rec", ".png", ".jpg"):
            viewer_imread(self.viewer, path)
        elif ext in (".csv", ".dat"):
            read_csv(self.viewer, path)
        elif ext in (".txt",):
            read_txt(self.viewer, path)
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
            return None
    
    def set_filter(self, names:str|list[str]):
        """
        Apply filter with comma separated string or list of string as an input.
        """        
        if isinstance(names, str):
            if names == "":
                names = "*"
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
    def selected(self) -> QModelIndex:
        inds = self.selectionModel().selectedIndexes()
        if len(inds) > 0:
            index = inds[0]
        else:
            index = None
        return index
