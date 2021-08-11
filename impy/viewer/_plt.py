import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backend_bases import MouseEvent
import warnings

__all__ = ["mpl", "plt", "canvas_plot", "EventedCanvas"]

class EventedCanvas(FigureCanvas):
    def __init__(self, fig):
        super().__init__(fig)
        self.pressed = False
        self.lastx = 0
        self.lasty = 0
        
    def wheelEvent(self, event):
        fig = self.figure
        try:
            ax = fig.axes[0]
        except IndexError:
            return None
        
        delta = event.angleDelta().y() / 120
        event = self.get_mouse_event(event, name="scroll_event")
        
        if not event.inaxes:
            return None
        
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
        fig.canvas.draw()
        return None
    
    def mousePressEvent(self, event):
        self.pressed = True
        event = self.get_mouse_event(event)
        self.lastx, self.lasty = event.xdata, event.ydata
        return None
        
    def mouseMoveEvent(self, event):
        if not self.pressed:
            return None
        fig = self.figure
        try:
            ax = fig.axes[0]
        except IndexError:
            return None
        event = self.get_mouse_event(event)
        
        if not event.inaxes:
            return None
        
        dx = event.xdata - self.lastx
        dy = event.ydata - self.lasty
        
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        
        ax.set_xlim([x0 - dx, x1 - dx])
        ax.set_ylim([y0 - dy, y1 - dy])
        fig.canvas.draw()
        return None

    def mouseReleaseEvent(self, event):
        self.pressed = False
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
        self.backend = mpl.get_backend()
        
        params = plt.style.library["dark_background"]
        mpl.rcParams.update(params)
        mpl.rcParams["figure.facecolor"] = "#0F0F0F"
        mpl.rcParams["axes.facecolor"] = "#0F0F0F"
        
        return self
    
    def __exit__(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            mpl.rcParams.update(self.original)