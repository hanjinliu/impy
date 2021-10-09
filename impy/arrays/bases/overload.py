from __future__ import annotations
import numpy as np
from .metaarray import MetaArray
from .historyarray import HistoryArray
from ...axes import Axes
from ...utils.axesop import del_axis
from ...collections import DataList

def safe_set_info(out, img, history, new_axes):
    if isinstance(img, HistoryArray):
        out._set_info(img, history, new_axes=new_axes)
    else:
        try:
            out._set_info(img, new_axes=new_axes)
        except Exception:
            pass
    return None


# Overloading numpy functions using __array_function__.
# https://numpy.org/devdocs/reference/arrays.classes.html


@MetaArray.implements(np.squeeze)
def _(img:MetaArray):
    out = np.squeeze(img.value).view(img.__class__)
    new_axes = "".join(a for a in img.axes if img.sizeof(a) > 1)
    safe_set_info(out, img, "squeeze", new_axes)
    return out

@MetaArray.implements(np.take)
def _(a:MetaArray, indices, axis=None, out=None, mode="raise"):
    new_axes = del_axis(a.axes, axis)
    if isinstance(axis, str):
        axis = a.axes.find(axis)
    out = np.take(a.value, indices, axis=axis, out=out, mode=mode).view(a.__class__)
    if isinstance(out, a.__class__):
        out._set_info(a, new_axes=new_axes)
    return out

@MetaArray.implements(np.stack)
def _(imgs:list[MetaArray], axis="c", dtype=None):
    """
    Create stack image from list of images.

    Parameters
    ----------
    imgs : iterable object of images.
        Images to stack. These images must have exactly the same shapes.
    axis : str or int, default is "c"
        Which axis will be the new one.
    dtype : str, optional
        Output dtype.

    Returns
    -------
    ImgArray
        Image stack
    """
    if imgs[0].axes:
        old_axes = str(imgs[0].axes)
    else:
        old_axes = None
    
    if isinstance(axis, int):
        _axis = axis
        axis = "p"
        if old_axes:
            new_axes = old_axes[:_axis] + axis + old_axes[_axis:]
        else:
            new_axes = None
    elif not isinstance(axis, str):
        raise TypeError(f"`axis` must be int or str, but got {type(axis)}")
    else:
        # find where to add new axis
        if old_axes:
            if imgs[0].axes.is_sorted():
                new_axes = Axes(axis + old_axes)
                new_axes.sort()
                _axis = new_axes.find(axis)
            else:
                new_axes = axis + old_axes
                _axis = 0
        else:
            new_axes = None
            _axis = 0

    if dtype is None:
        dtype = imgs[0].dtype

    arrs = [img.as_img_type(dtype).value for img in imgs]

    out = np.stack(arrs, axis=0)
    out = np.moveaxis(out, 0, _axis)
    out = out.view(imgs[0].__class__)
    safe_set_info(out, imgs[0], f"stack(axis={axis})", new_axes)
    return out

@MetaArray.implements(np.concatenate)
def _(imgs, axis="c", dtype=None, casting="same_kind"):
    if not isinstance(axis, (int, str)):
        raise TypeError(f"`axis` must be int or str, but got {type(axis)}")
    axis = imgs[0].axisof(axis)
    out = np.concatenate([img.value for img in imgs], axis=axis, dtype=dtype, casting=casting)
    out = out.view(imgs[0].__class__)
    safe_set_info(out, imgs[0], f"concatenate(axis={axis})", imgs[0].axes)
    return out

@MetaArray.implements(np.block)
def _(imgs):
    def _recursive_view(obj):
        if isinstance(obj, MetaArray):
            return obj.value
        else:
            return [_recursive_view(a) for a in obj]
    
    def _recursive_get0(obj):
        first = obj[0]
        if isinstance(first, MetaArray):
            return first
        else:
            return _recursive_get0(first)
    
    img0 = _recursive_get0(imgs)
    
    imgs = _recursive_view(imgs)
    out = np.block(imgs).view(img0.__class__)
    safe_set_info(out, img0, "block", img0.axes)
    return out


@MetaArray.implements(np.zeros_like)
def _(img, name:str=None):
    out = np.zeros_like(img.value).view(img.__class__)
    out._set_info(img, new_axes=img.axes)
    if isinstance(name, str):
        out.name = name
    return out

@MetaArray.implements(np.empty_like)
def _(img, name:str=None):
    out = np.empty_like(img.value).view(img.__class__)
    out._set_info(img, new_axes=img.axes)
    if isinstance(name, str):
        out.name = name
    return out

@MetaArray.implements(np.expand_dims)
def _(img, axis):
    if isinstance(axis, str):
        new_axes = Axes(axis + str(img.axes))
        new_axes.sort()
        axisint = tuple(new_axes.find(a) for a in axis)
    else:
        axisint = axis
        new_axes = img.axes
    
    out = np.expand_dims(img.value, axisint).view(img.__class__)
    safe_set_info(out, img, f"expand_dims({axis})", new_axes)
    return out

@MetaArray.implements(np.transpose)
def _(img, axes):
    return img.transpose(axes)

@MetaArray.implements(np.split)
def _(img, indices_or_sections, axis=0):
    if not isinstance(axis, (int, str)):
        raise TypeError(f"`axis` must be int or str, but got {type(axis)}")
    axis = img.axisof(axis)
    
    imgs = np.split(img.value, indices_or_sections, axis=axis)
    out = []
    for i, each in enumerate(imgs):
        each = each.view(img.__class__)
        safe_set_info(each, img, f"np.split[{i}]", new_axes="inherit")
        out.append(each)
    return DataList(out)