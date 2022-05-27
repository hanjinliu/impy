import impy as ip

def test_pointprops():
    img2d = ip.zeros((20, 20), axes="yx")
    img = ip.zeros((10, 20, 20), axes="tyx")
    assert img2d.pointprops([3, 3]) == 0.
    assert img.pointprops([3, 3]).shape == (10,)
    out = img.pointprops([[3, 3], [6, 3], [3, 6], [6, 6]])
    assert out.shape == (4, 10)  # N, t
