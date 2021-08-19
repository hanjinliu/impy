from __future__ import annotations
import napari
from qtpy.QtWidgets import QPlainTextEdit
from qtpy.QtGui import QFont
from qtpy.QtCore import Qt

from .textedit import WordHighlighter

class LoggerWidget(QPlainTextEdit):
    def __init__(self, viewer:"napari.Viewer"):
        super().__init__(viewer.window._qt_window)
        self.setReadOnly(True)
        self.setMaximumBlockCount(500)
        self.setFont(QFont("Consolas"))
        
        self.highlighter = WordHighlighter(self.document())
        self.highlighter.appendRule(r"[a-zA-Z]+Warning", fcolor=Qt.yellow)
        self.highlighter.appendRule(r"[a-zA-Z]+Error", fcolor=Qt.red)
    
    def appendPlainText(self, text:str):
        super().appendPlainText(text)
        self.highlighter.setDocument(self.document())
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        return None
    
    append = appendPlainText # compatibility with TableWidget
    
    def write(self, text:str):
        if text != "\n":
            self.appendPlainText(text)
        return None
        
    def flush(self): pass    # compatibility with IO
