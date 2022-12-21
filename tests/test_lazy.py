import impy as ip
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import pytest

filters = ["median_filter", "mean_filter", "erosion", "dilation", "opening", "closing",
           "kalman_filter", "gaussian_filter"]


def test_functions_and_slicing(resource):
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.lazy_imread(path, chunks=(4, 5, 2, 32, 32))
        sl = "y=20:40;x=30:50;c=0;z=2,4"
        assert_allclose(img[sl].compute(), img.compute()[sl])
        assert_allclose(
            img.affine(translation=[1, 50, 50]).compute(),
            img.compute().affine(translation=[1, 50, 50])
        )

@pytest.mark.parametrize("fn", filters)
def test_filters(fn, resource):
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.lazy_imread(path, chunks=(4, 5, 2, 32, 32))
        
        assert_allclose(
            getattr(img, fn)().compute(),
            getattr(img.compute(), fn)()
        )
        

def test_numpy_function():
    from dask.array.core import Array as DaskArray
    img = ip.aslazy(ip.random.random_uint16((10, 100, 100)))
    assert img.axes == "tyx"
    assert isinstance(np.mean(img).compute(), float)
    proj = np.mean(img, axis="y")
    assert isinstance(proj.value, DaskArray)
    assert proj.axes == "tx"
    assert isinstance(proj.compute(), ip.ImgArray)
    