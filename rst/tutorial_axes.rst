============
Axes in impy
============

.. contents:: Contents of Tutorial
    :local:
    :depth: 2

Basics
======

Understanding the concept of axes is important to use multi-dimensional arrays.

Each dimension of an array has its own meaning. A 3-D array can be composed of
three spatial axes or two spatial axes and a time axis.

You can create a ``ImgArray`` with any axes you want using ``axes`` argument.

.. code-block:: python

    import impy as ip
    img = ip.zeros((10, 128, 128), axes=["z", "y", "x"])

Any iterable objects can be used. In general, each axis label is represented by
a single character. In this case, it's simpler to use a ``str`` because a ``str``
itself is an iterable of ``str``.

.. code-block:: python

    img = ip.zeros((10, 128, 128), axes="zyx")

If you give axes of wrong length, or axes containing the same charactor, array
 creation will fail.

.. code-block:: python

    img = ip.zeros((10, 128, 128), axes="yx")  # Error!!
    img = ip.zeros((10, 128, 128), axes="tzyx")  # Error!!

Each axis is a ``Axis`` object. They are available by indexing ``Axes`` object.

.. code-block:: python

    img.axes  # Axes['z', 'y', 'x']
    img.axes[0]  # Axis['z']
    img.axes["y"]  # Axis['y']
    img.axes["a"]  # Error!


Undefined Axis
==============

Some functions and operations creates arrays with unknown axes.
In this case, ``UndefAxis`` objects are used for these axes and are represented by 
``"#"``.

.. code-block:: python

    img.axes  # Axes['z', 'y', 'x']
    np.expand_dims(img, axis=0).axes  # Axes['#', 'z', 'y', 'x']
    img[np.newaxis].axes  # Axes['#', 'z', 'y', 'x']
    img[img>0].axes  # Axes['#']
    img[[1, 2, 3], [2, 3, 4]].axes  # Axes['#', 'x']
    img.ravel().axes  # Axes["#"]

Axis Metadata
=============

Each axis could be tagged with some metadata. The major ones are physical scale,
physical scale unit and labels.

.. code-block:: python

    img.axes[0].scale  # scale of the first axis
    img.axes[0].unit  # scale unit of the first axis
    img.axes["c"].labels  # e.g. ("Red", "Green")
    img.axes[0].scale = 0.21
    img.axes[0].unit = "µm"

Physical scale
--------------

Physical scale is the length of value between ``a[i]`` and ``a[i+1]``. In image analysis,
this value is usually represented as "µm/pixel" or "nm/pixel" for spatial axes and "sec" for
time axis.

.. code-block:: python

    img.axes[0].scale  # scale of the first axis
    img.axes["x"].scale  # scale of x-axis
    img.axes[0].scale = 0.21  # update the scale of the first axis

You can refer to the scale unit with ``unit`` property.

.. code-block:: python

    img.axes[0].unit  # scale unit of the first axis
    img.axes[0].unit = "µm"  # update the scale unit

Since these values are tagged to ``Axis`` objects, they will be inherited after slicing,
filtering or any other operations.

.. code-block:: python

    img[0].axes["x"].scale == img.axes["x"].scale  # True
    img.gaussian_filter(sigma=1.0).axes["x"].scale == img.axes["x"].scale  # True
    (img + 1).axes["x"].scale == img.axes["x"].scale  # True
    np.mean(img, axis=0).axes["x"].scale == img.axes["x"].scale  # True

It is not always the case if you called certain functions that will change scales.

.. code-block:: python

    img[::2].axes[0].scale == img.axes[0].scale * 2  # True
    img[::-3].axes[0].scale == img.axes[0].scale * 3  # True
    img.binning(3) == img.axes[0].scale * 3

Axis Labels
-----------

Sometimes an axis is tagged with "labels" that explains what each slice means. ``Axis`` object
retains labels information and can be referred to as a tuple.

.. code-block:: python

    assert img.shape["t"] == 4  # say the length of t-axis is 4
    img.axes["t"].labels = ["0 sec", "10 sec", "30 sec", "1 min"]
    img.axes["t"].labels == ("0 sec", "10 sec", "30 sec", "1 min")

Because the length of labels must match corresponding shape of an array, it is safer to
use ``set_axis_label`` method. It checks the new labels.

.. code-block:: python

    img.set_axis_label(t=["0 sec", "10 sec", "30 sec", "1 min"])
    img.set_axis_label(t=["wrong", "input"])  # Error!

When array is sliced, labels are also correctly inherited

.. code-block:: python

    img.set_axis_label(t=["0 sec", "10 sec", "30 sec", "1 min"])
    img["t=:2"].axes["t"].labels == ("0 sec", "10 sec")  # True
    img["t=1,3"].axes["t"].labels == ("10 sec", "1 min")  # True

Practical Usage of Axes
=======================

Slicing and Formatting
----------------------

Axes object is very useful in slicing multi-dimensional arrays.

Axis-targeted slicing
^^^^^^^^^^^^^^^^^^^^^

As shown in tutorial, the easiest way to slice an array is to use axis axis-targeted slicing.

.. code-block:: python

    img["t=1"]
    img["t=3:5"]

This slicing method, however, ignores Python type-checking a little bit since you'll not notice
any wrong slicing grammar in the string until you run the code.

``impy`` also support a ``Slicer`` object for safer axis-targeted slicing.

.. code-block:: python

    ip.slicer.t[2].x[4:6]

.. code-block::
    
    Slicer of 
        t ==> 2
        x ==> 4:6

A ``Slicer`` object can be used for indexing an axis-implemented array.

.. code-block:: python

    img[ip.slicer.t[1]]  # equivalent to img["t=1"]
    img[ip.slicer.t[3:5]]  # equivalent to img["t=3:5"]
    img[ip.slicer.t[2, 4, 6]]  # equivalent to img["t=2,4,6"]
    img[ip.slicer.t[2].x[4]]  # equivalent to img["t=1;x=4"]

Slice Formatting
^^^^^^^^^^^^^^^^

Sometimes you would slice many times at the same axes.

.. code-block:: python

    img[ip.slicer.z[0].t[2]].gaussian_filter(1.0)
    img[ip.slicer.z[1].t[1]].gaussian_filter(1.5)
    img[ip.slicer.z[2].t[0]].gaussian_filter(1.0)

In this case, you can format slices using ``get_formatter`` method.

.. code-block:: python

    fmt = ip.slicer.get_formatter("zt")
    fmt

.. code-block::

    SliceFormatter of 
        z ==> Undefined
        t ==> Undefined


.. code-block:: python

    fmt[0, 2]

.. code-block::

    Slicer of 
        z ==> 0
        t ==> 2

Thus, you'll code will be 

.. code-block:: python

    img[fmt[0, 2]].gaussian_filter(1.0)
    img[fmt[1, 1]].gaussian_filter(1.5)
    img[fmt[2, 0]].gaussian_filter(1.0)

Broadcasting
------------

By using axes information, arrays can be broadcasted in a more flexible but strict way.

- Examples

    .. code-block:: python

        img0 = ip.random.random((12, 10, 14), axes="zyx")
        img1 = ip.random.random((12, 14), axes="zx") 
        
        np.asarray(img0) + np.asarray(img1)  # ValueError
        img0 + img1  # OK!

    .. code-block:: python

        img = ip.random.random((12, 12, 12), axes="tyx")
        img0 = np.mean(img, axis="y")  # axes: 't', 'x'
        img1 = np.mean(img, axis="x")  # axes: 't', 'y'
        np.asarray(img0) + np.asarray(img1)  # No error, but they should not be added!
        img0 + img1  # Error!

``impy`` also has a ``broadcast_arrays`` function for broadcasting arrays as flexible as
possible.

- Examples

    .. code-block:: python

        x = ip.arange(10, axes="x")
        y = ip.arange(8, axes="y")
        out = ip.broadcast_arrays(y, x)
        out[0].shape  # AxesShape(y=8, x=10)
        out[1].shape  # AxesShape(y=8, x=10)
    
    
    .. code-block:: python

        x = ip.random.random((5, 6, 7), axes="tzx")
        y = ip.random.random((4, 5, 7), axes="ntx")
        out = ip.broadcast_arrays(y, x)
        out[0].shape  # AxesShape(n=4, t=5, z=6, x=7)
        out[1].shape  # AxesShape(n=4, t=5, z=6, x=7)
