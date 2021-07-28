Tutorial
========

As a simple example:

.. code-block:: python

   import impy as ip
   img = ip.imread("path/to/image.tif")  # read
   out = img.gaussian_filter()           # process
   out

.. code-block::

        shape     : 10(t), 20(z), 256(y), 256(x)
      label shape : No label
        dtype     : uint16
      directory   : ...\images
    original image: XXX
        history   : gaussian_filter(sigma=1)

.. code-block:: python

    ip.gui.add(out)

and you can view the image in a ``napari`` viewer.

If images are saved as separate tif files in a directory,

.. code-block:: python

   img = ip.imread("path/to/image/*.tif")


Axes Targeted Slicing
---------------------

.. code-block:: python

    img["t=3;z=5:7"]
    img["y=3,5,7"] = 0

Batch Processing
----------------

Owing to the axes information, impy can automatically execute functions for every image slice properly.


For images with different shapes, you can use ``DataList``, an extension of Python ``list``. ``DataList`` recognizes any
member functions of its components and call the function for all the components. Here's an example:

.. code-block:: python

    imglist = ip.DataList([img1, img2, img3])
    outputs = imglist.gaussian_filter(sigma=3)

``gaussian_filter`` is a member function of ``img1``, ``img2`` and ``img3``, so that inside ``imglist``, ``gaussian_filter``
is called three times. Following code is essentially same as what is going on inside ``DataList``:

.. code-block:: python

    outputs = []
    for img in imglist:
        out = img.gaussian_filter(sigma=3)
        otuputs.append(out)
    outputs = ip.DataList(outputs)


Extended Numpy functions
------------------------

In almost all the ``numpy`` functions, the keyword argument ``axis`` can be given as the symbol of axis like:

.. code-block:: python

    np.mean(img, axis="z")           # Z-projection, although ImgArray provides more flexible function "proj()"
    np.stack([img1, img2], axis="c") # Merging colors

This is achieved by defining ``__array_function__`` method. See `here <https://numpy.org/devdocs/reference/arrays.classes.html>`_ for details.

You can also make an `ImgArray` in a way similar to ``numpy``:

.. code-block:: python

    ip.array([2, 4, 6], dtype="uint16")
    ip.zeros((100, 100), dtype=np.float32)
    ip.random.normal(size=(100, 100))


Read Very Large Images
----------------------

.. code-block:: python

    img = ip.lazy_imread("path/to/image/*.tif")
    img

.. code-block::
    
        shape     : 300(t), 25(z), 1024(y), 1024(x)
     chunk sizes  : 1(t), 25(z), 1024(y), 1024(x)
        dtype     : uint16
    directory   : ...\images
    original image: XXX
        history   : 

And check its size in GB:

.. code-block:: python

    img.gb

.. code-block::

    15.72864



