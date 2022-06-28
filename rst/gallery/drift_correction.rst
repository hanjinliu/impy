================
Drift Correction
================

Stage shift during image acquisition is a general problem of microscopy.
Here shows how to correct the shift using ``drift_correction`` method.

.. contents:: Contents
    :local:
    :depth: 1

Simple Example
==============

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

