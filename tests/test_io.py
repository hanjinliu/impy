import impy as ip
import tempfile
from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pytest

@pytest.mark.parametrize(
    ["ext", "unit"], 
    [(".tif", "μm"), (".mrc", "nm"), (".zarr", "μm")]
)
def test_imread_and_imsave(ext, unit):
    img = ip.random.random_uint16((4, 100, 100), axes="zyx")
    img.set_scale(z=0.4, xy=0.3)
    img.scale_unit = unit
    with tempfile.TemporaryDirectory() as path:
        file_path = Path(path) / f"test{ext}"
        img.imsave(file_path)
        img0 = ip.imread(file_path)
        assert img.dtype == img0.dtype == "uint16"
        assert img.axes == img0.axes
        assert_allclose(img.scale, img0.scale)
        assert img.scale_unit == img0.scale_unit
        assert_equal(img, img0)


path = Path(__file__).parent / "_test_images" / "image_tzcyx.tif"
img_orig = ip.imread(path)

@pytest.mark.parametrize(
    "key",
    ["y=:10;x=:10",
     "t=1;y=2,4,6",
     ip.slicer.y[10:].x[12:],
     ip.slicer.c[1].x[12:],
     ])
def test_imread_key(key):
    img0 = ip.imread(path, key=key)
    img1 = img_orig[key]
    assert_equal(img0, img1)
    assert img1.scale_unit == "μm"
    assert img0.scale_unit == "μm"

def test_imsave_safety():
    img = ip.random.random_uint16((4, 100, 100), axes="zyx")
    img.set_scale(z=0.4, xy=0.3)
    img.scale_unit = "μm"
    assert img.source is None
    with tempfile.TemporaryDirectory() as path:
        img.imsave(Path(path) / "test.tif")
        with pytest.raises(Exception):
            img.imsave("test.tif")
        img.source = Path(path) / "dummy.tif"
        img.imsave("test.tif")
        img.imsave("test")
        with pytest.raises(Exception):
            img.imsave("test.tif", overwrite=False)
