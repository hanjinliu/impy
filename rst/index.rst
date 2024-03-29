.. impy documentation master file, created by
   sphinx-quickstart on Mon Jul 26 15:21:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

impy
====

``impy`` is an all-in-one image analysis library, equipped with parallel processing, GPU support, GUI based tools and 
so on.

`Source code <https://github.com/hanjinliu/impy>`_


Highlights
----------

- Automatic parallel batch processing using ``dask``.

- You don't have to care about ``numpy`` / ``scipy`` on CPU, or ``cupy`` on GPU. Same code works for both processors.

- n-D viewing, cropping, image annotation using ``napari``.

- Easily integrate your custom functions with ``@ip.bind``.

- Command line usage.


Installation
------------

.. code-block:: shell

   pip install git+https://github.com/hanjinliu/impy


or

.. code-block:: shell

   git clone https://github.com/hanjinliu/impy

Contents
--------

.. toctree::
   :maxdepth: 1

   tutorial
   tutorial_axes
   tutorial_cmd
   ./gallery/index
   api
   

Major Classes
-------------

Array
^^^^^

.. blockdiag::
   
   blockdiag {
      orientation = portrait
      
      numpy.ndarray -> MetaArray -> LabeledArray -> ImgArray;
      AxesMixin -> MetaArray;
      MetaArray -> PropArray;
      MetaArray -> Label;
      LabeledArray -> PhaseArray;
      
      PropArray [color = pink];
      Label [color = pink];
      ImgArray [color = pink];
      PhaseArray [color = pink];
   }

- ``AxesMixin``: An abstract class that axes, scale and shape are defined.

- ``PropArray``: Array object with properties stored in it. Always made from an ``ImgArray``.

- ``Label``: Array object of image labels that is attached to ``ImgArray``.

- ``ImgArray``: Array object with many image processing functions.

- ``PhaseArray``: Array object with periodic values and specific processing functions.


Array-like
^^^^^^^^^^

.. blockdiag::
   
   blockdiag {
      orientation = portrait

      AxesMixin -> LazyImgArray -> BigImgArray;
      LazyImgArray [color = pink];
      BigImgArray [color = pink];
   }

- ``LazyImgArray``: Array-like object with image processing functions like ``ImgArray``, but evaluated lazily.
- ``BigImgArray``: Subclass of ``LazyImgArray``, but images will be processed for every method call.


Data Frame  
^^^^^^^^^^

.. blockdiag::
   
   blockdiag {
      orientation = portrait

      pandas.DataFrame -> AxesFrame -> MarkerFrame;
      AxesFrame -> TrackFrame;
      AxesFrame -> PathFrame;
      
      MarkerFrame [color = pink];
      TrackFrame [color = pink];
      PathFrame [color = pink];
   }

- ``AxesFrame``: DataFrame with similar properties as ``AxesMixin``.

- ``MarkerFrame``: ``AxesFrame`` for markers, such as coordinates.

- ``TrackFrame``: ``AxesFrame`` for tracks.

- ``PathFrame``: ``AxesFrame`` for paths.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
