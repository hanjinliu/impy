from __future__ import annotations
import napari
from qtpy.QtWidgets import QPlainTextEdit
from qtpy.QtGui import QFont

class LoggerWidget(QPlainTextEdit):
    def __init__(self, viewer:"napari.Viewer"):
        super().__init__(viewer.window._qt_window)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setFont(QFont("Consolas"))
    
    def appendPlainText(self, text:str):
        super().appendPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        return None
    
    append = appendPlainText # compatibility with TableWidget
    write = appendPlainText  # compatibility with IO
    def flush(self): pass    # compatibility with IO
