================
Drift Correction
================

Stage shift during image acquisition is a general problem of microscopy.
Here shows how to correct the shift using ``drift_correction`` method.

.. contents:: Contents
    :local:
    :depth: 1


Simplest Correction
===================

We'll create a sample image stack by applying random shifts to an image using ``affine`` method
and Gaussian noise using ``ip.random.normal`` function.

.. code-block:: python
    
    import numpy as np
    import impy as ip

    t_total = 10
    max_shift = 6
    img0 = ip.zeros((64, 64))
    img0[28:36, 22:30] += 1.0

    imgs = []
    shifts = (2 * np.random.random((t_total, 2)) - 1.0) * max_shift  # random shift
    for shift in shifts:
        img_shift = img0.affine(translation=shift) + ip.random.normal(scale=0.3, size=(64, 64)) 
        imgs.append(img_shift)

    img = np.stack(imgs, axis="t")

The ``img`` is a 3-D image stack with a randomly shifting rectangle in the center.

``drift_correction`` method uses phase cross-correlation to track drift and restore non-drifted
image by Affine transformation. Relative shift between neighbors are calculated.

It is very simple to obtain the corrected image.

.. code-block:: python

    img_corrected = img.drift_correction()
    img_corrected = img.drift_correction(along="t")  # explicitly specify the "time" axis.
    
    img_corrected.imshow()

Multi-dimensional Correction
============================

By default, ``drift_correction`` consider ``"y", "x"`` axes as the spatial dimensions and conduct
2-D correction unlike many of other methods. This is because image drift usually occurs in XY
direction. The example below:

.. code-block:: python

    img4d = ip.random.normal(size=(10, 4, 64, 64), axes="tzyx")
    img_corrected = img4d.drift_correction()

is almost equal to:

.. code-block:: python

    out = []
    for z in range(4):
        out.append(img4d[:, z].drift_correction())
    img_corrected = np.stack(out, axis="z")

.. note::

    ``drift_correct`` uses the most plausible axis as the "time" axis. To avoid unexpected error
    you should specify ``along`` argument when correction >4 dimensional images.

Correction with Reference
=========================

Sometimes you may want to supply a "reference" image stack to determine drift, instead of using 
the image itself. There are many occasions that you should think of this.

- Multi-channel image. In most cases, image shifts are the same among all the channels. You
  may want to choose one of the channels (a channel that is the most static) for tracking.
- Images with strong noises. You should not use the region of an image stack if that region 
  contains such kind of noises. For instance, cropping the image at its edges like
  ``img["y=40:-40;x=40:-40"]`` will be helpful.
- Use calculated images. A simple case is to use a filtrated reference image stack.

``drift_correction`` takes ``ref`` argument to do this. If ``ref`` is given, ``drift_correction``
checks dimensionalities of the image to correct and the reference image and can flexibly apply
tracking and correction.

.. code-block:: python

    # Use the first channel to track drift of a multi-channel image
    img = ip.random.normal((10, 3, 64, 64), axes="tcyx")
    img_corrected = img.drift_correction(ref=img["c=0"])

    # Use the center of an image
    img = ip.random.normal((10, 8, 64, 64), axes="tzyx")
    img_corrected = img.drift_correction(ref=img["y=40:-40;x=40:-40"])

    # Use Gaussian-filtrated image
    img = ip.random.normal((10, 8, 64, 64), axes="tzyx")
    img_corrected = img.drift_correction(ref=img.gaussian_filter())

Correct Large Images
====================

TODO