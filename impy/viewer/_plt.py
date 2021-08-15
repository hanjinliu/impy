from functools import wraps
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvas
from matplotlib.backend_bases import MouseEvent, MouseButton, FigureManagerBase

__all__ = ["mpl", "plt_figure", "EventedCanvas", "StyledFigure"]

# add new style
params = plt.style.library["dark_background"]
params["figure.facecolor"] = "#0F0F0F"
params["axes.facecolor"] = "#0F0F0F"
params["font.size"] = 10.5
params["grid.color"] = "gray"
plt.style.library["night"] = params

class EventedCanvas(FigureCanvas):
    """
    A figure canvas implemented with mouse callbacks.
    """    
    def __init__(self, fig):
        super().__init__(fig)
        self.pressed = None
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
                factor = 1/1.3
            else:
                factor = 1.3

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
            self.pressed = event.button
        return None
        
    def mouseMoveEvent(self, event):
        """
        Translate axes focus while dragging. If there are subplots, only the subplot in which
        cursor exists will be translated.
        """        
        if self.pressed not in (MouseButton.LEFT, MouseButton.RIGHT) or self.lastx is None:
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
            if self.pressed is MouseButton.LEFT:
                ax.set_xlim([x0 - dx, x1 - dx])
                ax.set_ylim([y0 - dy, y1 - dy])
            elif self.pressed is MouseButton.RIGHT:
                ax.set_xlim([x0, x1 - dx])
                ax.set_ylim([y0, y1 - dy])
            break

        fig.canvas.draw()
        return None

    def mouseReleaseEvent(self, event):
        """
        Stop dragging state.
        """        
        self.pressed = None
        return None
    
    def mouseDoubleClickEvent(self, event):
        """
        Adjust layout upon dougle click.
        """        
        self.figure.tight_layout()
        self.figure.canvas.draw()
        return None
    
    def resizeEvent(self, event):
        """
        Adjust layout upon canvas resized.
        """        
        super().resizeEvent(event)
        self.figure.tight_layout()
        self.figure.canvas.draw()
        return None
    
    def get_mouse_event(self, event, name="") -> MouseEvent:
        x, y = self.mouseEventCoords(event)
        if hasattr(event, "button"):
            button = self.buttond.get(event.button())
        else:
            button = None
        mouse_event = MouseEvent(name, self, x, y, button=button, guiEvent=event)
        return mouse_event
        
    
def change_style(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        with plt.style.context("night"):
            out = func(self, *args, **kwargs)
        return out
    return wrapped_func

def plt_figure(num=None, figsize=None, dpi=None, frameon=True, clear=False, **kwargs):
    """
    Call ``plt.figure`` in the context of using StyledFigure.
    """    
    return plt.figure(facecolor=params["figure.facecolor"], FigureClass=StyledFigure, 
                      num=num, figsize=figsize, dpi=dpi, frameon=frameon, clear=clear, **kwargs)

class StyledFigure(mpl.figure.Figure):
    @change_style
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @change_style
    def add_subplot(self, *args, **kwargs):
        return super().add_subplot(*args, **kwargs)
    
    @change_style
    def subplots(self, *args, **kwargs):
        return super().subplots(*args, **kwargs)
    
    @change_style
    def legend(self, *args, **kwargs):
        return super().legend(*args, **kwargs)
        

class NapariFigureManager(FigureManagerBase):
    """
    A figure manager that enables plotting inside the napari viewer.
    """    
    def __init__(self, canvas=None, num=1):
        from . import gui
        canvas = gui.fig.canvas
        gui.fig.clf()
        super().__init__(canvas, num)
        
    def show(self):
        self.canvas.draw()
        return None

def new_figure_manager(*args, **kwargs):
    """
    Always use StyledFigure class.
    """    
    kwargs["FigureClass"] = StyledFigure
    return NapariFigureManager()