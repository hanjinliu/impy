Viewer Tutorial
===============

Send Data from Console to Viewer
--------------------------------

You can add any objects (images, labels, points, ...) to the viewer by ``ip.gui.add(...)``. ``ip.gui`` can determine layer types according to
the instance type.

1. Add images

When ``ImgArray``, ``PhaseArray``, ``LazyImgArray``, or path to the image file are given, an ``Image`` layer with proper settings will be created.
If ``np.ndarray`` or ``dask.array.core.Array`` are given, they will be converted to plausible array types that are compatible with ``impy`` and
similarly send to the viewer.

.. code-block:: python

    # Basically you'll run ...
    img = ip.imread("path/to/img.tif")
    ip.gui.add(img)
    # but this also works
    ip.gui.add("path/to/img.tif")

If ``LazyImgArray`` is given, or a path is given but the image size is too large, then image is loaded as ``dask`` array so that it will be viewed
as "virtual stack", i.e., read from the disk every time it is needed to. However, viewing large images is not as slow as you expect because image 
data will be cached by methods in ``dask.cache``.

2. Add labels

When ``Label`` or ``ImgArray`` that has ``labels`` attribute are given, a ``Labels`` layer will be created.

3. Add points

When ``MarkerFrame`` or ``TrackFrame`` are given, a ``Points`` layer will be created.

.. code-block:: python

    mols = img.find_sm()    # find single molecules
    ip.gui.add(img)         # add image
    ip.gui.add(mols)        # add points

4. Add table widgets

When ``PropArray`` or ``pandas.DataFrame`` are given, an Excel-like table widget will be added on the right side of the viewer.

Get impy Objects from Viewer
----------------------------

The ``napari.Viewer`` object is accessible via ``ip.gui.viewer``, so that basically you can call any method from it. However, 

- Return all the manually selected layers' data by ``layers = ip.gui.selection``.
- Run ``ImgArray``'s method inside viewers.

Mouse Callbacks
---------------

- ``Alt`` + mouse drag -> lateral translation
- ``Alt`` + ``Shift`` + mouse drag -> lateral translation restricted in either x- or y-orientation (left button or right button respectively).
- ``Alt`` + mouse wheel -> rescaling
- ``Ctrl`` + ``Shift`` + ``R`` -> reset original states.


Keyboard Shortcuts
------------------

- ``Ctrl`` + ``Shift`` + ``A`` -> Hide non-selected layers. Display all the layers by push again.
- ``Ctrl`` + ``Shift`` + ``F`` -> Move selected layers to front.
- ``Alt`` + ``L`` -> Convert all the shapes in seleted shape-layers into labels of selected image-layers.
- ``Ctrl`` + ``Shift`` + ``D`` -> Duplicate selected layers.
- ``Ctrl`` + ``Shift`` + ``X`` -> Crop selected image-layers with all the rectangles in selected shape-layers. Rotated cropping is also supported!
- ``/`` -> Reslice selected image-layers with all the lines and paths in selected shape-layers. Result is stored in ``ip.gui.results`` for now.
- ``Ctrl`` + ``P`` -> Projection of shape-layers or point-layers to 2D layers.
- ``Ctrl`` + ``G`` / ``Ctrl`` + ``Shift`` + ``G`` -> Link/Unlink layers. Like "grouping" in PowerPoint.
- ``S`` -> Add nD shape-layer.
- ``P`` -> Add nD point-layer.

Others
------

- Show coordinates of selected point-layers or track-layers. You can also copy it to clipboard.
- Note pad in ``Window > Note``.
- Call ``impy.imread`` in ``File > imread ...``. Call ``impy.imsave`` in ``File > imsave ...``.