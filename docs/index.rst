.. impy documentation master file, created by
   sphinx-quickstart on Mon Jul 26 15:21:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to impy's documentation!
================================

``impy`` is a fast to code, easy to extend image processing package. 

Highlights
----------

- Automatic parallel batch processing using ``dask``.
- You don't have to care about ``numpy`` / ``scipy`` on CPU, or ``cupy`` on GPU. Same code works for both processors.
- n-D viewing, cropping, image annotation using ``napari``.
- Easily integrate your custom functions with ``@ip.bind``.
- Easily make a interactive GUI plugin from your custom functions with ``@ip.gui.bind``

`Source code <https://github.com/hanjinliu/impy>`_


Installation
------------

.. code-block:: shell

   pip install git+https://github.com/hanjinliu/impy


or

.. code-block:: shell

   git clone https://github.com/hanjinliu/impy


Contents
--------

.. blockdiag::

   blockdiag {
      default_fontsize = 24;

      numpy.ndarray -> MetaArray -> HistoryArray -> LabeledArray -> ImgArray;
      AxesMixin -> MetaArray;
      MetaArray -> PropArray;
      HistoryArray -> Label;
      LabeledArray -> PhaseArray;
   }


.. toctree::
   :maxdepth: 1

   tutorial
   tutorial_viewer
   api
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
