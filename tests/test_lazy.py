import tempfile
import impy as ip
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
import pytest
import operator

filters = [
    "median_filter",
    "mean_filter",
    "erosion",
    "dilation",
    "opening",
    "closing",
    "kalman_filter",
    "gaussian_filter",
    "edge_filter",
]


def test_functions_and_slicing(resource):
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.lazy.imread(path, chunks=(4, 5, 2, 32, 32))
        sl = "y=20:40;x=30:50;c=0;z=2,4"
        assert_allclose(img[sl].compute(), img.compute()[sl])
        assert_allclose(
            img.affine(translation=[1, 50, 50]).compute(),
            img.compute().affine(translation=[1, 50, 50])
        )

@pytest.mark.parametrize("fn", filters)
def test_filters(fn, resource):
    if resource == "cupy" and fn in ("edge_filter",):
        # not supported on GPU
        return
    with ip.SetConst(RESOURCE=resource):
        path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
        img = ip.lazy.imread(path, chunks=(4, 5, 2, 32, 32))

        assert_allclose(
            getattr(img, fn)().compute(),
            getattr(img.compute(), fn)()
        )


def test_numpy_function():
    from dask.array.core import Array as DaskArray
    rng = ip.lazy.random.default_rng(0)
    img = rng.random_uint16((2, 3, 4))
    assert img.axes == "tyx"
    assert isinstance(np.mean(img).compute(), float)
    proj = np.mean(img, axis="y")
    assert isinstance(proj.value, DaskArray)
    assert proj.axes == "tx"
    assert isinstance(proj.compute(), ip.ImgArray)


@pytest.mark.parametrize("opname", ["gt", "lt", "ge", "le", "eq", "ne"])
def test_operator(opname):
    op = getattr(operator, opname)
    rng = ip.lazy.random.default_rng(0)
    img1 = rng.random_uint16((2, 3, 4))
    img2 = rng.random_uint16((2, 3, 4))
    assert_allclose(
        op(img1, img2).compute(),
        op(img1.compute(), img2.compute()),
    )

@pytest.mark.parametrize("ext", [".tif", ".mrc"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_lazy_imsave(ext: str, dtype):
    rng = ip.random.default_rng(1234)
    if dtype == np.uint16:
        img = rng.random_uint16((4, 8, 80), axes="zyx")
    else:
        img = rng.random((4, 8, 80), axes="zyx")
    img_lazy = ip.lazy.asarray(img, chunks=(1, 2, 80))
    img_lazy.scale_unit = "nm"
    with tempfile.TemporaryDirectory() as path:
        fp = Path(path) / f"test{ext}"
        img_lazy.imsave(fp)
        img0 = ip.imread(fp)
        assert img_lazy.shape == img0.shape
        assert img_lazy.dtype == img0.dtype
        assert img_lazy.axes == img0.axes
        assert_allclose(img, img0)
