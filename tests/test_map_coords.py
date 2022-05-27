from numpy.testing import assert_allclose
import impy as ip
from scipy import ndimage as ndi

def test_map_coordinates():
    img = ip.arange(120, dtype=ip.float32).reshape(10, 12, axes="yx")
    
    coords = ip.random.random((2, 4, 4), axes="duv") * 10
    ans = ndi.map_coordinates(img.value, coords.value)
    
    out = img.map_coordinates(coords)
    assert out.axes == ["u", "v"]
    assert_allclose(ans, out)
    
    out = img.map_coordinates(coords.value)
    assert out.axes == ["y", "x"]
    assert_allclose(ans, out)
    
    coords = ip.random.random((2, 7), axes="dv") * 10
    ans = ndi.map_coordinates(img.value, coords.value)
    
    out = img.map_coordinates(coords)
    assert out.axes == ["v"]
    assert_allclose(ans, out)
    
    out = img.map_coordinates(coords.value)
    assert out.axes == ["#"]
    assert_allclose(ans, out)

def test_multi_map_coordinates():
    img = ip.arange(120 * 21, dtype=ip.float32).reshape(3, 7, 10, 12, axes="ctyx")
    
    coords = ip.random.random((2, 4, 4), axes="duv") * 10
    
    out = img.map_coordinates(coords)
    assert out.axes == ["c", "t", "u", "v"]
    
    out = img.map_coordinates(coords.value)
    assert out.axes == ["c", "t", "y", "x"]
    
    coords = ip.random.random((2, 7), axes="dv") * 10
    out = img.map_coordinates(coords)
    assert out.axes == ["c", "t", "v"]
    
    out = img.map_coordinates(coords.value)
    assert out.axes == ["c", "t", "#"]
