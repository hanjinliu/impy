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


Practical Usage of Axes
=======================

Slicing and Formatting
----------------------

TODO


Broadcasting
------------

TODO
