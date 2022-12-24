Tutorial
========

.. contents:: Contents of Tutorial
    :local:
    :depth: 2

Basics
------

``impy``'s concept is to make image analysis on Python much more user friendly. The core array object, 
``ImgArray``, retains all the features of ``numpy.ndarray`` while implemented with variety of function
that minimize your effort of image processing.

Let's start with a simple example.

.. code-block:: python

   import impy as ip
   img = ip.imread("path/to/image.tif")  # read
   out = img.gaussian_filter(sigma=1)    # process
   out

.. code-block::

    ImgArray of
         name      : image.tif
        shape      : 10(t), 20(z), 256(y), 256(x)
     label shape   : No label
        dtype      : uint16
        source     : path/to/image.tif
        scale      : ScaleView(t=1.0, z=0.217, y=0.217, x=0.217)
        
Here, you can see 

- each dimension is labeled with a symbol (t, z, y and x).
- Physical scale is tagged to the object as a ``ScaleView``, a subclass of Python ``dict``.
- the source file is recorded.

``impy``'s ``imread`` function can extract metadata (such as axes) from image files and "memorize" them 
as additional properties of array objects.

You can refer to the properties by:

.. code-block::

    img.axes       # Out: Axes['t', 'z', 'y', 'x']
    img.source     # Out: such as WindowsPath("path/to/image.tif")
    img.name       # Out: "image.tif"
    img.scale      # Out: ScaleDict(t=1.0px, z=0.217μm, y=0.217μm, x=0.217μm)

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

Similarly, image shape is also extended to support axis-based access. In the example above, the image of
interest has (t, z, y, x) axes and the shape is (10, 20, 256, 256). Similar to ``numpy.ndarray``, you can
know its shape using ``img.shape`` but in ``impy`` an object of ``AxesShape`` will be returned instead of
Python ``tuple``, as long as axes are well-defined.

.. code-block:: python

    img.shape       # Out: AxesShape(t=10, z=20, y=256, z=256)

Here, you can get the size of z-axis by ``img.shape.z`` instead of ``img.shape[1]``.

.. note::

    You can also use ``dict`` for slicing.
    
    .. code-block:: python

        img[{"y": 3, "x": slice(4, 10)}]  # identical to img["y=3;x=4:10"]

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
    out = img.gaussian_filter(sigma=1, dims=2)  # this is fine


Running Function with Different Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. Apply a function to whole image with different parameters

.. code-block:: python

    out = img.for_params("log_filter", var={"sigma": [1, 2, 3, 4]})
    out = img.for_params("log_filter", sigma=[1, 2, 3, 4]) # This is also supported.

2. Apply a function along an axis with different parameters

You usually want to apply same function to each channel but with different parameters.

.. code-block:: python

    out = img.for_each_channel("hessian_eigval", sigma=[1, 2])


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


Use GPU
-------

``impy`` can automatically switch between ``numpy`` and ``cupy``. Using GPU can largely boost
your image analysis especially when it relies on Fourier transformation or linear algebra.
You can setup GPU calculation within a context using

.. code-block:: python
    
    with ip.use("cupy"):
        img_deconv = img.lucy(psf_image)

or globally

.. code-block:: python
    
    ip.Const["RESOURCE"] = "cupy"


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


Large Images
------------

There are two ways to handle large images.

LazyImgArray
^^^^^^^^^^^^

If you deal with very large images that exceeds PC memory, you can use ``LazyImgArray``. This object retains
memory map of the image file that is split into smaller chunks, and passes it to ``dask`` array as "ready to
read" state. The image data is therefore loaded only when it is needed. Many useful functions in ``ImgArray`` 
are also implemented in ``LazyImgArray`` so that you can easily handle large datasets.

To read large images as ``LazyImgArray``, call ``lazy_imread`` instead. You can specify its chunk size using
``chunks`` parameter.

.. code-block:: python

    img = ip.lazy_imread("path/to/image.tif", chunks=(1, "auto", "auto", "auto"))
    img

.. code-block::
    
    LazyImgArray of
         name     : image.tif
        shape     : 300(t), 25(z), 1024(y), 1024(x)
     chunk sizes  : 1(t), 25(z), 1024(y), 1024(x)
        dtype     : uint16
        source    : path/to/image.tif
        scale     : ScaleView(t=1.0px, z=0.217μm, y=0.217μm, x=0.217μm)

You can check its size in GB:

.. code-block:: python

    img.GB

.. code-block::

    15.72864

When you have to convert it to ``ImgArray``, use ``compute`` function:

.. code-block:: python

    img.compute()  # dask's compute() function will be called inside

BigImgArray
^^^^^^^^^^^

``LazyImgArray`` is useful to process large images. However, it is not suitable for interactive analysis 
because calculation starts from the beginning for every operation. ``BigImgArray`` is a subclass of 
``LazyImgArray`` but it stores the cashed data in a temporary file.

You can use :meth:`big_imread` function to open an image file as a ``BigImgArray`` object.

.. code-block:: python

    img = ip.big_imread("path/to/image.tif")
    img

.. code-block::
    
    BigImgArray of
         name     : image.tif
        shape     : 300(t), 25(z), 1024(y), 1024(x)
     chunk sizes  : 1(t), 25(z), 1024(y), 1024(x)
        dtype     : uint16
        source    : path/to/image.tif
        scale     : ScaleView(t=1.0px, z=0.217μm, y=0.217μm, x=0.217μm)

And all the methods supported in ``LazyImgArray`` are available.

.. code-block:: python

    img1 = img.gaussian_filter()  # computed and cached here
    img2 = img1.threshold(img1.mean())  # computed and cached here
    out = img2.compute()  # convert to ImgArray
