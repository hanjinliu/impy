from __future__ import annotations
from qtpy.QtWidgets import QWidget, QTableWidget, QTableWidgetItem, QAbstractItemView, QAbstractScrollArea, QVBoxLayout
from qtpy.QtCore import Qt
import napari

class ResultStackView(QWidget):
    MAX_ROW_NUMBER = 50
    def __init__(self, viewer:"napari.Viewer"):
        super().__init__(viewer.window._qt_window)
        self.viewer = viewer
        self.setLayout(QVBoxLayout())
        self._list = []
        self._add_table()        
    
    def __len__(self):
        return self.table.rowCount()
    
    def __getitem__(self, key):
        return self._list[key]
    
    def __iter__(self):
        return iter(self._list)
    
    def __repr__(self):
        names = []
        for x in self._list:
            s = repr(x)
            if len(s) < 42:
                names.append(s)
            else:
                names.append(s.__class__.__name__)
        return f"{self.__class__.__name__} with \n{names}"
    
    def append(self, result):
        self._list.append(result)
        nrow = self.table.rowCount()
        
        self.table.insertRow(nrow)
        
        item = QTableWidgetItem(result.__class__.__name__)
        item.setFlags(Qt.ItemIsEnabled)
        self.table.setItem(nrow, 0, item)
        item = QTableWidgetItem(_as_short_str(result))
        item.setFlags(Qt.ItemIsEnabled)
        self.table.setItem(nrow, 1, item)
        
        if nrow > self.__class__.MAX_ROW_NUMBER:
            self.pop(0)
        
        # scroll to the last
        item = self.table.item(nrow, 0)
        self.table.scrollToItem(item, hint=QAbstractItemView.PositionAtTop)
        self.table.resizeColumnsToContents()
        
        if len(self._list) == 1:
            from ..viewer import ResultsWidgetName
            self.viewer.window._dock_widgets[ResultsWidgetName].show()
        return None
    
    def pop(self, i:int=-1):
        if i < 0:
            i = len(self) + i
        out = self._list.pop(i)
        self.table.removeRow(i)
        return out
    
    def clear(self):
        self.layout().removeWidget(self.table)
        self._list = []
        self._add_table()
        return None

    def _add_table(self):
        self.table = QTableWidget(0, 3, self)
        self.table.setHorizontalHeaderItem(0, QTableWidgetItem("Type"))
        self.table.setHorizontalHeaderItem(1, QTableWidgetItem("Value"))
        self.table.setHorizontalHeaderItem(2, QTableWidgetItem("Note"))
        self.table.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
        self.table.resizeColumnsToContents()
        self.layout().addWidget(self.table)
        return None

def _as_short_str(obj):
    try:
        s = str(obj)
    except Exception:
        s = obj.__class__.__name__
    if "\n" in s:
        i = s.find("\n")
        s = s[:min(i, 20)] + " ..."
    
    elif len(s) > 20:
        s = s[:20] + " ..."
    return s