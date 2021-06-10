import napari
import magicgui
from qtpy.QtWidgets import QFileDialog, QAction, QPushButton, QWidget, QGridLayout
from .utils import *

__all__ = ["add_imread_menu", "add_table_widget", "add_note_widget"]

def add_imread_menu(viewer):
    from ..core import imread
    def open_img():
        dlg = QFileDialog()
        hist = napari.utils.history.get_open_history()
        dlg.setHistory(hist)
        filenames, _ = dlg.getOpenFileNames(
            parent=viewer.window.qt_viewer,
            caption='Select file ...',
            directory=hist[0],
        )
        if (filenames != []) and (filenames is not None):
            img = imread(filenames[0])
            add_labeledarray(viewer, img)
        napari.utils.history.update_open_history(filenames[0])
        return None
    
    viewer.window.file_menu.addSeparator()
    action = QAction('imread ...', viewer.window._qt_window)
    action.triggered.connect(open_img)
    viewer.window.file_menu.addAction(action)
    return None



def add_table_widget(viewer):
    QtViewerDockWidget = napari._qt.widgets.qt_viewer_dock_widget.QtViewerDockWidget
    
    button = QPushButton("Get")
    @button.clicked.connect
    def make_table():
        dfs = list(iter_selected_layer(viewer, ["Points", "Tracks"]))
        if len(dfs) == 0:
            return
        for df in dfs:
            widget = QWidget()
            widget.setLayout(QGridLayout())
            columns = list(df.metadata["axes"])
            table = magicgui.widgets.Table(df.data, name=df.name, columns=columns)
            copy_button = QPushButton("Copy")
            copy_button.clicked.connect(lambda: table.to_dataframe().to_clipboard())                    
            widget.layout().addWidget(table.native)
            widget.layout().addWidget(copy_button)
            
            widget = QtViewerDockWidget(viewer.window.qt_viewer, widget, name=df.name,
                                        area="right", add_vertical_stretch=True)
            viewer.window._add_viewer_dock_widget(widget, tabify=viewer.window.n_table>0)
            viewer.window.n_table += 1
        return None
    
    viewer.window.add_dock_widget(button, area="left", name="Get Coordinates")
    viewer.window.n_table = 0
    return None


def add_note_widget(viewer):
    text = magicgui.widgets.TextEdit(tooltip="Note")
    text = viewer.window.add_dock_widget(text, area="right", name="Note")
    text.setVisible(False)
    return None