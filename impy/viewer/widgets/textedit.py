from __future__ import annotations
import os
import re
import napari
from qtpy.QtWidgets import (QPlainTextEdit, QWidget, QLineEdit, QGridLayout, QLabel, QHBoxLayout, QCheckBox,
                            QMenuBar, QMenu, QAction)
from qtpy.QtGui import QFont, QSyntaxHighlighter, QTextCharFormat
from qtpy.QtCore import QRegularExpression, Qt

def read_txt(viewer:"napari.Viewer", path:str):
    with open(path, mode="r") as f:
        content = f.read()
    title, _ = os.path.splitext(os.path.basename(path))
    text = TxtFileWidget(viewer, title=title)
    text.initText(content)
    return viewer.window.add_dock_widget(text, area="right", name=title)

class TxtFileWidget(QWidget):
    """
    A read-only text viewer widget with JSON-like highlight. Capable of search lines.
    """
    def __init__(self, viewer:"napari.Viewer", title:str=None):
        super().__init__(viewer.window._qt_window)
        self.viewer = viewer
        self.title = title
        self.setLayout(QGridLayout())
        self._add_menu()
        self._add_txt_viewer(title)
        self._add_filter_line()
    
    def initText(self, text:str):
        self.original_text = text
        self.setText(text)
        
    def setText(self, text:str):
        self.txtviewer.setPlainText(text)
        self.highlighter.setDocument(self.txtviewer.document())
    
    def _add_menu(self):
        self.menubar = QMenuBar(self)
        self.menu = QMenu("&Menu", self)
        
        wrap = QAction("Wrap", self, checkable=True, checked=False)
        wrap.triggered.connect(self.change_wrap_mode)
        self.menu.addAction(wrap)
        
        close = QAction("Close", self)
        close.triggered.connect(self.delete_self)
        self.menu.addAction(close)
        
        self.menubar.addMenu(self.menu)
        
        self.layout().addWidget(self.menubar)
        return None
        
    def _add_txt_viewer(self, title):
        self.txtviewer = TxtViewer(self, title=title)
        self.txtviewer.setReadOnly(True)
        self.txtviewer.setFont(QFont("Consolas"))
        title is None or self.txtviewer.setDocumentTitle(title)
        self.highlighter = WordHighlighter(self.txtviewer.document())
        self.highlighter.appendRule(r" [-\+]?\d*(\.\d+)?", fcolor=Qt.yellow)
        self.highlighter.appendRule(r"\".*?\"", fcolor=Qt.green)
        self.layout().addWidget(self.txtviewer)
    
    def _add_filter_line(self):
        wid = QWidget(self)
        wid.setLayout(QHBoxLayout())
        
        # add label
        label = QLabel(self)
        label.setText("Search:")
        
        # add line edit
        self.line = QLineEdit(self)
        self.line.setToolTip("Search line by line with words or regular expressions.")
        @self.line.editingFinished.connect
        def _():
            text = self.line.text()
            if text in ("", ".", ".+", ".*"):
                if self.highlighter.nrule > 2:
                    self.highlighter.popRule(-1)
                self.setText(self.original_text)
            else:
                lines = self.original_text.split("\n")
                if self.checkbox.isChecked():
                    reg = re.compile(".*" + text + ".*")
                    matched_lines = filter(lambda l: reg.match(l), lines)
                else:
                    matched_lines = filter(lambda l: text in l, lines)
                
                if self.highlighter.nrule > 2:
                    self.highlighter.popRule(-1)    
                self.highlighter.appendRule(text, fcolor=Qt.cyan, underline=True, weight=QFont.Bold)
                self.setText("\n".join(matched_lines))
        
        # add check box
        self.checkbox = QCheckBox(self)
        self.checkbox.setText("Regex")
        self.checkbox.setToolTip("Use regular expression.")
        
        wid.layout().addWidget(label)
        wid.layout().addWidget(self.line)
        wid.layout().addWidget(self.checkbox)
        
        self.layout().addWidget(wid)
    
    def change_wrap_mode(self):
        # line wrap mode = 0 -> No wrap 
        # line wrap mode = 1 -> wrapped
        mode = self.txtviewer.lineWrapMode()
        self.txtviewer.setLineWrapMode(1-mode)
        return None
    
    def delete_self(self):
        dock = self.viewer.window._dock_widgets[self.title]
        self.viewer.window.remove_dock_widget(dock)
        return None

class TxtViewer(QPlainTextEdit):
    def __init__(self, parent, title:str=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas"))
        self.createStandardContextMenu()
        self.setLineWrapMode(0)
        
        title is None or self.setDocumentTitle(title)

class WordHighlighter(QSyntaxHighlighter):
    def __init__(self, doc) -> None:
        super().__init__(doc)
        self.exps:list[Regex] = []
    
    def appendRule(self, regex:str, **kwargs):
        exp = Regex(regex)
        exp.defineFormat(**kwargs)
        self.exps.append(exp)
    
    def popRule(self, i:int):
        return self.exps.pop(i)
    
    @property
    def nrule(self):
        return len(self.exps)
        
    def highlightBlock(self, text:str):
        for exp in self.exps:
            it = exp.globalMatch(text)
            while it.hasNext():
                match = it.next()
                self.setFormat(match.capturedStart(), match.capturedLength(), exp.format_)


class Regex(QRegularExpression):
    def defineFormat(self, fcolor=None, bcolor=None, weight=None, underline=False, strikeout=False):
        format_ = QTextCharFormat()
        fcolor is None or format_.setForeground(fcolor)
        bcolor is None or format_.setBackground(bcolor)
        weight is None or format_.setFontWeight(weight)
        underline and format_.setFontUnderline(True)
        strikeout and format_.setFontStrikeOut(True)
        self.format_ = format_
        return None