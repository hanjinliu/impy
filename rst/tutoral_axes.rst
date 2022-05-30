============
Axes in impy
============

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

Broadcasting
------------

TODO
