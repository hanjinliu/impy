Tutorial
========

.. contents:: Contents of Tutorial
    :local:
    :depth: 2

Basics
------

Let's start with a simple example.

.. code-block:: python

   import impy as ip
   img = ip.imread("path/to/image.tif")  # read
   out = img.gaussian_filter(sigma=1)    # process
   out

.. code-block::

        shape     : 10(t), 20(z), 256(y), 256(x)
      label shape : No label
        dtype     : uint16
      directory   : path/to
    original image: image
        history   : gaussian_filter(sigma=1)

Here, you can see 

- each dimension is labeled with a symbol (t, z, y and x)
- the source directory and file names are recorded
- history of image analysis is recorded

``impy``'s ``imread`` function can extract metadata (such as axes) from image files and "memorize" them 
as additional properties of array objects. Most of array objects in ``impy`` can record history, which
makes it much easier to review what changes were applied to the images.

You can refer to the properties by:

.. code-block::

    img.axes        # Out: "tzyx"
    img.dirpath     # Out: "path/to"
    img.name        # Out: "image"
    img.history     # Out: ["gaussian_filter(sigma=1)"]

After you finished all the process you want, you can view the results in ``napari`` viewer. Just pass the
image object to GUI handler object ``gui`` in ``impy``.

.. code-block:: python

    ip.gui.add(out)

and wait until viewer window opens.


Axes Targeted Slicing
---------------------

You may want to get slice(s) of an image with a format like "t=3:10" but generally you always have to
care about which is t-axis. Most arrays in ``impy`` have extended ``numpy.ndarray`` to enable the
"axis-targeted slicing". You can use following grammar:

- Single slice, like ``img["t=1"]``.
- String that follows Python slicing rules, such as ``img["t=5:10"]`` or ``img["t=-1"]``
- Fancy slicing, like ``img["t=1,3,5"]``
- Conbination of them with splitter ";", like ``img["t=3;z=5:7"]``


Batch Processing
----------------

Image Stacks
^^^^^^^^^^^^

Owing to the axes information, impy can automatically execute functions for every image slice properly.
As in the first example, with a `tzyx` image, instead of running

.. code-block:: python

    out = np.empty_like(img)
    for t in range(10):
        out[t] = img[t].gaussian_filter(sigma=1)

you just need to run a single code

.. code-block:: python

    out = img.gaussian_filter(sigma=1)

and the function "knows" `zyx` or `(1,2,3)` axes are spatial dimensions and filtering should be iterated along `t` axis.

If you want `yx` axes be the spatial dimensions, i.e., iterate over `t` and `z` axes, explicitly specify it with ``dims``
keyword argument:

.. code-block:: python

    out = img.gaussian_filter(sigma=1, dims="yx")
    out = img.gaussian_filter(sigma=1, dims=2) # this is fine!


Images with Different Shapes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For images with different shapes, they cannot be stacked into a single array. In this case, you can use ``DataList``, an 
extension of Python ``list``. ``DataList`` recognizes any member functions of its components and call the function for all 
the components. Here's an example:

.. code-block:: python

    imglist = ip.DataList([img1, img2, img3])
    outputs = imglist.gaussian_filter(sigma=3)

``gaussian_filter`` is a member function of ``img1``, ``img2`` and ``img3``, so that inside ``imglist``, ``gaussian_filter``
is called three times. Following code is essentially same as what is going on inside ``DataList``:

.. code-block:: python

    outputs = []
    for img in imglist:
        out = img.gaussian_filter(sigma=3)
        outputs.append(out)
    outputs = ip.DataList(outputs)

``impy`` also provides ``DataDict``, an extension of Python ``dict``, which works similarly to ``DataList``. Aside from
the feature of iterative function call, you can give names for each image as dictionary keys, and get the value from 
attribution, ``imgdict.name`` instead of ``imgdict["name"]``.

.. code-block:: python

    imglist = ip.DataDict(first=img1, second=img2, third=img3)
    outputs = imglist.gaussian_filter(sigma=3)
    outputs.first


Extended Numpy functions
------------------------

In almost all the ``numpy`` functions, the keyword argument ``axis`` can be given as the symbol of axis if the argument(s) are ``ImgArray`` 
or other arrays that belong to subclass of ``MetaArray``.

.. code-block:: python

    np.mean(img, axis="z")           # Z-projection, although ImgArray provides more flexible function "proj()"
    np.stack([img1, img2], axis="c") # Merging colors

This is achieved by defining ``__array_function__`` method. See `Numpy's documentation <https://numpy.org/devdocs/reference/arrays.classes.html>`_ 
for details.

You can also make an `ImgArray` in a way similar to ``numpy``:

.. code-block:: python

    ip.array([2, 4, 6], dtype="uint16")
    ip.zeros((100, 100), dtype=np.float32)
    ip.random.normal(size=(100, 100))


Advanced Reading Options
------------------------

Read Separate Images as an Image Stack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If images are saved as separate tif files in a directory, you can read them as an image stack by:

.. code-block:: python

   img = ip.imread("path/to/image/*.tif")


Read Separate Images as an DataList
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   img = ip.imread_collection("path/to/image/*.tif")


Read Very Large Images
^^^^^^^^^^^^^^^^^^^^^^

If you deal with very large images that exceeds PC memory, you can use ``LazyImgArray``. This object retains
memory map of the image file that is split into smaller chunks, and passes it to ``dask`` array as "ready to
read" state. The image data is loaded only when it is needed. Many useful functions in ``ImgArray`` are also
implemented in ``LazyImgArray`` so that you can easily handle large datasets.

To read large images as ``LazyImgArray``, call ``lazy_imread`` instead.

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

You can check its size in GB:

.. code-block:: python

    img.gb

.. code-block::

    15.72864



