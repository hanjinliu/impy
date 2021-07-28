Viewer Tutorial
===============

- Add any objects (images, labels, points, ...) to the viewer by ``ip.gui.add(...)``.
- Return all the manually selected layers' data by ``layers = ip.gui.selection``.
- Run ``ImgArray``'s method inside viewers.

Mouse Callbacks
---------------

- ``Alt`` + mouse drag &rarr; lateral translation
- ``Alt`` + ``Shift`` + mouse drag &rarr; lateral translation restricted in either x- or y-orientation (left button or right button respectively).
- ``Alt`` + mouse wheel &rarr; rescaling
- ``Ctrl`` + ``Shift`` + ``R`` &rarr; reset original states.


Keyboard Shortcuts
------------------

- ``Ctrl`` + ``Shift`` + ``A`` &rarr; Hide non-selected layers. Display all the layers by push again.
- ``Ctrl`` + ``Shift`` + ``F`` &rarr; Move selected layers to front.
- ``Alt`` + ``L`` &rarr; Convert all the shapes in seleted shape-layers into labels of selected image-layers.
- ``Ctrl`` + ``Shift`` + ``D`` &rarr; Duplicate selected layers.
- ``Ctrl`` + ``Shift`` + ``X`` &rarr; Crop selected image-layers with all the rectangles in selected shape-layers. Rotated cropping is also supported!
- ``/`` &rarr; Reslice selected image-layers with all the lines and paths in selected shape-layers. Result is stored in ``ip.gui.results`` for now.
- ``Ctrl`` + ``P`` &rarr; Projection of shape-layers or point-layers to 2D layers.
- ``Ctrl`` + ``G`` / ``Ctrl`` + ``Shift`` + ``G`` &rarr; Link/Unlink layers. Like "grouping" in PowerPoint.
- ``Shift`` + ``S`` / ``S`` &rarr; Add 2D/nD shape-layer.
- ``Shift`` + ``P`` / ``P`` &rarr; Add 2D/nD point-layer.

Others
------

- Show coordinates of selected point-layers or track-layers. You can also copy it to clipboard.
- Note pad in ``Window > Note``.
- Call ``impy.imread`` in ``File > imread ...``. Call ``impy.imsave`` in ``File > imsave ...``.