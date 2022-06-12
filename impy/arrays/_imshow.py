from __future__ import annotations
from .labeledarray import LabeledArray
from ..utils.axesop import find_first_appeared

class ImshowManager:
    def __init__(self):
        self._mgr = {}
    
    def register(self, name: str):
        def _f(f):
            self._mgr[name] = f
        return _f
    
    def call(self, *args, plugin="matplotlib", **kwargs):
        return self._mgr[plugin](*args, **kwargs)
    
MANAGER = ImshowManager()

imshow = MANAGER.call

@MANAGER.register("magicgui")
def _imshow_magicgui(arr: LabeledArray, label, dims, **kwargs) -> LabeledArray:
    from magicgui.widgets import Image, Slider, Container
    
    if arr.ndim == 2:
        widget = Image(value=arr)
        widget.min_height = 300
    else:
        if "c" in arr.axes:
            ...
        img = Image(value=arr[(0,) * (arr.ndim - 2)])
        img.min_height = 300
        sliders = Container(
            widgets=[Slider(max=arr.shape[a], name=str(a)) for a in arr.axes if a not in dims]
        )
        sliders.margins = (0, 0, 0, 0)
        @sliders.changed.connect
        def _on_slider_change():
            vals = tuple(sl.value for sl in sliders)
            img.value = arr[vals]
            
        widget = Container(widgets=[img, sliders])
    
    widget.show(kwargs.get("run", False))
    return arr

@MANAGER.register("matplotlib")
def _imshow_matplotlib(self: LabeledArray, label, dims, **kwargs) -> LabeledArray:
    from ._utils import _plot as _plt
    alpha = 0.3
    if label and self.labels is None:
        label = False
    if self.ndim == 1:
        _plt.plot_1d(self.value, **kwargs)
    elif self.ndim == 2:
        if label:
            _plt.plot_2d_label(self.value, self.labels.value, alpha, **kwargs)
        else:
            _plt.plot_2d(self.value, **kwargs)
        self.hist()
        
    elif self.ndim == 3:
        if "c" not in self.axes:
            imglist = self.split(
                axis=find_first_appeared(self.axes, include=self.axes, exclude=dims)
            )
            if len(imglist) > 24:
                import warnings
                warnings.warn(
                    "Too many images. First 24 images are shown.",
                    UserWarning,
                )
                imglist = imglist[:24]
            if label:
                _plt.plot_3d_label(imglist.value, imglist.labels.value, alpha, **kwargs)
            else:
                _plt.plot_3d(imglist, **kwargs)

        else:
            n_chn = self.shape.c
            fig, ax = _plt.subplots(1, n_chn, figsize=(4*n_chn, 4))
            for i in range(n_chn):
                img = self[f"c={i}"]
                if label:
                    _plt.plot_2d_label(img.value, img.labels.value, alpha, ax[i], **kwargs)
                else:
                    _plt.plot_2d(img.value, ax=ax[i], **kwargs)
    else:
        raise ValueError("Image must have three or less dimensions.")
    
    _plt.show()

    return self