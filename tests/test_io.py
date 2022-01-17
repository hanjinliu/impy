import impy as ip
import tempfile
from pathlib import Path
from numpy.testing import assert_allclose, assert_equal

def test_tif():
    img = ip.random.random_uint16((8, 8, 8), axes="zyx")
    img.set_scale(z=0.4, xy=0.3)
    img.scale_unit = "um"
    with tempfile.TemporaryDirectory() as path:
        file_path = Path(path)/"test.tif" 
        img.imsave(file_path)
        img0 = ip.imread(file_path)
        assert img.dtype == img0.dtype == "uint16"
        assert img.axes == img0.axes
        assert_allclose(img.scale, img0.scale)
        assert img.scale_unit == img0.scale_unit
        assert_equal(img, img0)
    
def test_mrc():
    img = ip.random.random_uint16((8, 8, 8), axes="zyx")
    img.set_scale(z=0.4, xy=0.3)
    with tempfile.TemporaryDirectory() as path:
        file_path = Path(path)/"test.mrc" 
        img.imsave(file_path)
        img0 = ip.imread(file_path)
        assert img.dtype == img0.dtype == "uint16"
        assert img.axes == img0.axes
        assert_allclose(img.scale, img0.scale)
        assert img.scale_unit == img0.scale_unit
        assert_equal(img, img0)
    
        