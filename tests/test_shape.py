import impy as ip

def test_shapes():
    shape = (4, 5, 6)
    img = ip.zeros(shape, axes="zyx")
    assert img.shape == shape
    assert img.shape.z == img.shape["z"] == shape[0]
    assert img.shape.y == img.shape["y"] == shape[1]
    assert img.shape.x == img.shape["x"] == shape[2]
    
    shape = (9, 3, 12)
    img = ip.zeros(shape, axes="zyx")
    assert img.shape == shape
    assert img.shape.z == img.shape["z"] == shape[0]
    assert img.shape.y == img.shape["y"] == shape[1]
    assert img.shape.x == img.shape["x"] == shape[2]
