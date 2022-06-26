=============
Image Filters
=============

There are several filtering methods implemented in ``ImgArray``.

.. code-block:: python

    import impy as ip
    
    img = ip.random.random((5, 32, 64, 64), axes="tzyx")
    img

.. code-block::

    ImgArray of
          name      : random
         shape      : 5(t), 32(z), 64(y), 64(x)
      label shape   : No label
         dtype      : float32
         source     : None
         scale      : ScaleView(t=1.0, z=1.0, y=1.0, x=1.0)


.. contents:: Contents
    :local:
    :depth: 2

Batch Processing
----------------

By default, ``ImgArray`` consider ``["z", "y", "x"]`` axes as spatial axes and iterate functions
along other axes. In this example, 3-D filter will be applied for every ``"t"``. If you want 
other iteration options, explicitly specify ``dims`` keyword argument.

.. code-block:: python

    img.gaussian_filter(sigma=2.0)
    img.gaussian_filter(sigma=2.0, dims="yx")  # ["y", "x"] is considered as spatial axes.
    img.gaussian_filter(sigma=2.0, dims=["y", "x"])  # same as dims="yx".

Denoising
---------

Gaussian filter
^^^^^^^^^^^^^^^

Gaussian filter is a widely used denoising filter. It blurs image using Gaussian kernel.
``sigma`` is standard deviation (in pixel) of the kernel.

.. code-block:: python

    img.gaussian_filter()  # sigma=1.0 by default
    img.gaussian_filter(sigma=2.0)  # use sigma=2.0
    img.gaussian_filter(sigma=[2.0, 1.0, 1.0])  # non-uniform sigma

Median filter
^^^^^^^^^^^^^

Median filter is a denoising that is considered to be robust against outliers. Kernel shape is
specified by ``radius`` argument.

.. code-block:: python

    img.median_filter()  # radius=1.0 by default
    img.median_filter(radius=3.2)
    img.median_filter(radius=[3.2, 1.0, 1.0])  # non-uniform radii

Mean filter
^^^^^^^^^^^

Mean filter (uniform filter) is a simple denoising, where image is locally averaged with same
weight.

.. code-block:: python

    img.mean_filter()  # radius=1.0 by default
    img.mean_filter(radius=3.2)
    img.mean_filter(radius=[3.2, 1.0, 1.0])  # non-uniform radii

Use Standard Deviation
----------------------

Standard deviation filter
^^^^^^^^^^^^^^^^^^^^^^^^^

Standard deviation filter can detect regions that signal changes a lot.

.. code-block:: python

    img.std_filter()  # radius=1.0 by default
    img.std_filter(radius=3.2)
    img.std_filter(radius=[3.2, 1.0, 1.0])  # non-uniform radii


Coefficient of variation filter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Coefficient of variation is a quantity that is defined by `S.D. / mean`. Coefficient of
variation filter is similar to standard deviation filter but is not sensitive to mean intensity. 

.. code-block:: python

    img.coef_filter()  # radius=1.0 by default
    img.coef_filter(radius=3.2)
    img.coef_filter(radius=[3.2, 1.0, 1.0])  # non-uniform radii

Feature Detection
-----------------

Edge detection
^^^^^^^^^^^^^^

Edge detection filters generate images that have large value at the regions that signal change
largely. You can consider them as scalar differentiation of images. Different edge detection
filter used slightly different kernel but these kernels always take positive values on one side
while take negative on the other.

.. code-block:: python

    img.edge_filter()  # Sobel filter by default
    img.edge_filter(method="farid")  # Farid filter
    img.edge_filter(method="scharr")  # Scharr filter
    img.edge_filter(method="prewitt")  # Prewitt filter

Puncta detection
^^^^^^^^^^^^^^^^

Puncta detection filters are useful for automatic molecule detection with images taken by light
or electron microscope. Note that images must be dark-background.

.. code-block:: python

    img.dog_filter()  # DoG (Difference of Gaussian)
    img.doh_filter()  # DoH (Difference of Hessian)
    img.log_filter()  # Log (Laplacian of Gaussian)

Filament detection
^^^^^^^^^^^^^^^^^^

A "filament" can be defined by 2nd derivative: convex in one direction and flat in the perpendicular
direction. This trick can be achieved by inspecting the Hessian of an image.

``hessian_eigval`` is composed of two steps. First, apply Gaussain filter to the image. Then,
eigenvalues of Hessian are calculated. That's why it has ``sigma`` argument.

.. code-block:: python

    vals = img.hessian_eigval()
    vals = img.hessian_eigval(sigma=2.0)

The returned array has a new axis named ``"base"``, which corresponds to each spatial axis.

.. code-block::

    ImgArray of
          name      : random
         shape      : 3(base), 5(t), 32(z), 64(y), 64(x)
      label shape   : No label
         dtype      : float32
         source     : None
         scale      : ScaleView(base=1.0, t=1.0, z=1.0, y=1.0, x=1.0)
