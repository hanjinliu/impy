import impy as ip
import tempfile
from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pytest

@pytest.mark.parametrize(
    ["ext", "unit"], 
    [(".tif", "um"), (".mrc", "nm"), (".zarr", "um")]
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
