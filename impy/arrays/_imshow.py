from __future__ import annotations
from typing import Callable
import numpy as np

from impy.arrays.labeledarray import LabeledArray
from impy.utils.axesop import find_first_appeared, complement_axes
from impy.axes import slicer

__all__ = ["imshow", "register"]

class ImshowManager:
    """Manage imshow plugins."""
    
    def __init__(self):
        self._mgr: dict[str, Callable] = {}
    
    def register(self, name: str):
        def _f(f):
            if name in self._mgr:
                raise ValueError(f"Plugin {name!r} already exists.")
            self._mgr[name] = f
        return _f
    
    def call(self, *args, plugin="matplotlib", **kwargs):
        if plugin not in self._mgr:
            raise ValueError(f"Plugin {plugin!r} not found.")
        return self._mgr[plugin](*args, **kwargs)
    
MANAGER = ImshowManager()

imshow = MANAGER.call
register = MANAGER.register

# ################################################################################
#   magicgui
# ################################################################################

@register("magicgui")
def _imshow_magicgui(arr: LabeledArray, dims, **kwargs) -> LabeledArray:
    from magicgui import widgets as wdt

    h, w = arr.sizesof(dims)
    if h > w:
        min_height = 400
        min_width = 400 / h * w
    else:
        min_height = 400 / w * h
        min_width = 400
    
    min_, max_ = arr.range
    if arr.ndim == 1:
        raise NotImplementedError()
    if arr.ndim == 2:
        value = arr.value
        img = wdt.Image(value=value)
        sliders = wdt.EmptyWidget(visible=False)
    else:
        c_axes = complement_axes(dims, arr.axes)
        fmt = slicer.get_formatter(c_axes)
        img = wdt.Image(value=arr[fmt.zeros()])
        sliders = wdt.Container(
            widgets=[wdt.Slider(max=arr.shape[a] - 1, name=str(a)) for a in c_axes]
        )
        sliders.margins = (0, 0, 0, 0)
        @sliders.changed.connect
        def _on_slider_change():
            vals = tuple(sl.value for sl in sliders)
            img_slice = arr[fmt[vals]]
            img.value = img_slice

    clim = wdt.FloatRangeSlider(min=min_, max=max_, value=(min_, max_), step=(max_ - min_) / 1000, name="clim")
    imgc = wdt.Container(widgets=[img])
    imgc.min_height = min_height
    imgc.min_width = min_width
    imgc.margins = (0, 0, 0, 0)
    if arr.labels is not None:
        cbox = wdt.CheckBox(text="Show labels", value=kwargs.get("label", False))
        alpha = wdt.FloatSlider(name="Label alpha", value=0.3, min=0.0, max=1.0)
        label_wdt = wdt.Container(widgets=[cbox, alpha])
        
        @cbox.changed.connect
        @alpha.changed.connect
        @sliders.changed.connect
        @clim.changed.connect
        def _on_slider_change():
            vals = tuple(sl.value for sl in sliders)
            img_slice = arr[fmt[vals]]
            if cbox.value:
                from skimage.color import label2rgb

                vmin, vmax = clim.value
                image = (np.clip(img_slice, vmin, vmax) - vmin)/(vmax - vmin)
                img.value = label2rgb(
                    img_slice.labels, image, alpha=alpha.value, bg_label=0, image_alpha=1,
                    bg_color=None,
                )
            else:
                img.value = img_slice
            img.set_clim(*clim.value)

    else:
        label_wdt = wdt.EmptyWidget()
        
        @sliders.changed.connect
        def _on_slider_change():
            vals = tuple(sl.value for sl in sliders)
            img_slice = arr[fmt[vals]]
            img.value = img_slice
        
        @clim.changed.connect
        def _on_clim_change():
            img.set_clim(*clim.value)
    
    widget = wdt.Container(
        widgets=[imgc, sliders, clim, label_wdt], 
        labels=False,
    )
    
    if "cmap" in kwargs:
        img.set_cmap(kwargs["cmap"])
    widget.show(kwargs.get("run", False))
    return arr

# ################################################################################
#   matplotlib
# ################################################################################

@register("matplotlib")
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


@register("ipywidgets")
def _imshow_ipywidgets(arr: LabeledArray, label: bool, dims, **kwargs) -> LabeledArray:
    from ipywidgets import interact
    
    raise NotImplementedError()