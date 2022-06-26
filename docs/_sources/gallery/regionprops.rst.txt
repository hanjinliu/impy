========================
Labeling and Measurement
========================

Here we're going to label a image and measure some features for each label.

.. contents:: Contents
    :local:
    :depth: 1

Basics
------

Get the sample image.

.. code-block:: python

    import impy as ip
    img = ip.sample_image("coins")

There are two methods for image labeling.

- ``label`` ... labeling image with the input reference.
- ``label_threshold`` ... labeling image with the binarized image as the reference. It can be 
  considered as a shortcut of ``img.label(img.threshold())``.

Both method returns a ``Label`` array, a subclass of ``MetaArray``.

Simple labeling
---------------

.. code-block:: python

    img.label_threshold()

.. code-block::

    Label of
          name      : coins
         shape      : 303(y), 384(x)
         dtype      : uint8
         source     : None
         scale      : ScaleView(y=1.0, x=1.0)

The ``Label`` is tagged to the image itself.

.. code-block:: python

    img.labels

.. code-block::

    Label of
          name      : coins
         shape      : 303(y), 384(x)
         dtype      : uint8
         source     : None
         scale      : ScaleView(y=1.0, x=1.0)

Slicing of image covariantly slices the labels at the same time.

.. code-block:: python

    img[100:200, 53].labels

.. code-block::

    Label of
          name      : coins
         shape      : 100(y)
         dtype      : uint8
         source     : None
         scale      : ScaleView(y=1.0)

Image with label overlay can be visualized by ``imshow`` method or using ``napari`` viewer.

.. code-block:: python

    img.imshow(label=True)  # use matplotlib to show images
    ip.gui.add(img)  # use napari to show images

Measurement
-----------

After labeling, ``regionprops`` method is useful for image measurement. This method runs 
``skimage.measure.regionprops`` inside.

.. code-block:: python

    props = img.regionprops(properties=("mean_intensity", "area", "major_axis_length"))
    props

.. code-block::
    
    DataDict[PropArray] with 3 components:
    'mean_intensity' => PropArray of
          name      : coins-prop
         shape      : 98(N)
         dtype      : float32
         source     : None
     property name  : mean_intensity
    ,
    'area' => PropArray of
          name      : coins-prop
         shape      : 98(N)
         dtype      : float32
         source     : None
     property name  : area
    ,
    'major_axis_length' => PropArray of
          name      : coins-prop
         shape      : 98(N)
         dtype      : float32
         source     : None
     property name  : major_axis_length

The returned ``DataDict`` object is a ``dict``-like object. Its value is assured to be the same
type so that you can easily apply a same method to all the components (see :doc:`../tutorial.rst`).
Since "mean_intensity", "area" and "major_axis_length" are chosen for measurement, ``props`` has
keys "mean_intensity", "area" and "major_axis_length".

Here, ``props`` is a ``DataDict`` of ``PropArray``. ``PropArray`` is a subclass of ``MetaArray``
that is specialized in storing properties.

All the properties can be summerized as follows.

.. code-block:: python

    # since PropArray has method `mean`, this line will apply `mean` to all the components.
    props.mean()

.. code-block::

    DataDict[float32] with 3 components:
    'mean_intensity' => 122.47181,
    'area' => 465.52042,
    'major_axis_length' => 15.488672

.. code-block:: python

    # PropArray has a visualization method `hist`.
    props.hist()

Conditional Labeling
--------------------

Simple labeling based on thresholding always yields in insufficient results. 

The ``filt`` argument can filter labels based on properties of image and labels.
Basic usage is following. Filter function must take at least two argument, image itself and
newly created label region.

.. code-block:: python

    def filt(img, lbl):
        """Return true if label passes a criterion you set."""
    
    img.label_threshold(filt=filt)

You can use additional arguments with names same as those properties supported in
``regionprops``. For instance, you can label regions only satisfies proper area and length 
using following filter function.

.. code-block:: python

    def filt(img, lbl, area, major_axis_length):
        proper_size = 10 < area < 60**2
        proper_shape = 20 < major_axis_length < 120
        return proper_size and proper_shape
    
    img.label_threshold(filt=filt)
