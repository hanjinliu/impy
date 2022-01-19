Command Line Usage
==================

Image analysis usually relies on different softwares in different platform.
The traditional command line based interface is still useful in this purpose.

``impy`` also supports command line usage.

Basic Usage
-----------

The most basic one is:

.. code-block:: shell

    impy --input path/to/image.tif --output path/to/output.tif --method gaussian_filter --sigma 2

or ``--input`` and ``--output`` can be omitted like:

.. code-block:: shell

    impy path/to/image.tif path/to/output.tif --method gaussian_filter --sigma 2

You have to specify ``--method`` or ``-m`` to let ``impy`` know what method you'd like to run.
Method names should match those in ``ImgArray``. The last ``--sigma`` option is characteristic 
to ``gaussian_filter`` so that it may differ with other method.
Those commands above are equivalent to following Python code.

.. code-block:: python

    import impy as ip
    img = ip.imread("path/to/image.tif")
    out = img.gaussian_filter(sigma=2)
    out.imsave("path/to/output.tif")


Advanced Usage
--------------

Sometimes you may want to apply multiple filters in tandem to an image. Classically it was
usually done by saving intermediate files, but here we should take advantage of IPython.

Instead of parsing ``--output``, you can use ``-i`` flag to launch IPython interpreter with
namespace ``ip`` and ``img``. ``ip`` is an alias of ``impy`` as usual, and ``img`` is an
``ImgArray`` object that is created by reading the ``--input`` argument.

.. code-block:: shell

    impy path/to/image.tif -i

.. code-block:: python

    >>> arr = ip.random.random((100, 120))  # "ip" is already available
    >>> out = img + arr  # img = ip.imread("path/to/image.tif")
    >>> out.imsave("path/to/output.tif")

Another option is ``-n``, which send namespace including input image to ``napari`` viewer
and console.

.. code-block:: shell

    impy path/to/image.tif -n
