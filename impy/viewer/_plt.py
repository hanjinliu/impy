import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backend_bases import MouseEvent, FigureManagerBase
import warnings

__all__ = ["mpl", "plt", "canvas_plot", "EventedCanvas"]

# TODO: resize xlim/ylim separately when outside is clicked

class EventedCanvas(FigureCanvas):
    """
    A figure canvas implemented with mouse callbacks.
    """    
    def __init__(self, fig):
        super().__init__(fig)
        self.pressed = False
        self.lastx = None
        self.lasty = None
        
    def wheelEvent(self, event):
        """
        Resize figure by changing axes xlim and ylim. If there are subplots, only the subplot
        in which cursor exists will be resized.
        """        
        fig = self.figure
        
        delta = event.angleDelta().y() / 120
        event = self.get_mouse_event(event)
        
        if not event.inaxes:
            return None
        for ax in fig.axes:
            if event.inaxes != ax:
                continue
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            
            if delta > 0:
                factor = 1/1.1
            else:
                factor = 1.1

            ax.set_xlim([(x1 + x0)/2 - (x1 - x0)/2*factor,
                         (x1 + x0)/2 + (x1 - x0)/2*factor])
            ax.set_ylim([(y1 + y0)/2 - (y1 - y0)/2*factor,
                         (y1 + y0)/2 + (y1 - y0)/2*factor])
            break
        fig.canvas.draw()
        return None
    
    def mousePressEvent(self, event):
        """
        Record the starting coordinates of mouse drag.
        """        
        event = self.get_mouse_event(event)
        self.lastx, self.lasty = event.xdata, event.ydata
        if event.inaxes:
            self.pressed = True
        return None
        
    def mouseMoveEvent(self, event):
        """
        Translate axes focus while dragging. If there are subplots, only the subplot in which
        cursor exists will be translated.
        """        
        if not self.pressed or self.lastx is None:
            return None
        fig = self.figure
        
        event = self.get_mouse_event(event)
        
        for ax in fig.axes:
            if event.inaxes != ax:
                continue
            dx = event.xdata - self.lastx
            dy = event.ydata - self.lasty
            
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            
            ax.set_xlim([x0 - dx, x1 - dx])
            ax.set_ylim([y0 - dy, y1 - dy])
            break

        fig.canvas.draw()
        return None

    def mouseReleaseEvent(self, event):
        """
        Stop dragging state.
        """        
        self.pressed = False
        return None
    
    def mouseDoubleClickEvent(self, event):
        """
        Adjust layout upon dougle click.
        """        
        self.figure.tight_layout()
        self.figure.canvas.draw()
        return None
    
    def get_mouse_event(self, event, name="") -> MouseEvent:
        x, y = self.mouseEventCoords(event)
        mouse_event = MouseEvent(name, self, x, y, event)
        return mouse_event
        
    
class canvas_plot:
    def __init__(self):
        pass
    
    def __enter__(self):
        self.original = mpl.rcParams.copy()
        
        params = plt.style.library["dark_background"]
        mpl.rcParams.update(params)
        mpl.rcParams["figure.facecolor"] = "#0F0F0F"
        mpl.rcParams["axes.facecolor"] = "#0F0F0F"
        mpl.rcParams["font.size"] = 10.5
        
        return self
    
    def __exit__(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mpl.rcParams.update(self.original)


class NapariFigureManager(FigureManagerBase):
    """
    A figure manager that enables plotting inside the napari viewer.
    """    
    def __init__(self, canvas=None, num=1):
        from . import gui
        canvas = gui.fig.canvas
        super().__init__(canvas, num)
        gui.fig.clf()
        
    def show(self):
        self.canvas.draw()
        return None

def new_figure_manager(*args, **kwargs):
    return NapariFigureManager()