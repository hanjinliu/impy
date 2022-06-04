import impy as ip

def test_remove_edges():
    img = ip.zeros((50, 50, 50), axes="zyx")
    img0 = img.remove_edges(3)
    assert img0.shape == (50, 44, 44)
    assert img0.axes == ["z", "y", "x"]
    
    img0 = img.remove_edges(3, dims="zyx")
    assert img0.shape == (44, 44, 44)
    assert img0.axes == ["z", "y", "x"]
    
    img0 = img.remove_edges([3, 5, 0], dims="zyx")
    assert img0.shape == (44, 40, 50)
    assert img0.axes == ["z", "y", "x"]
    

def test_crop_center_even():
    img = ip.zeros((60, 60, 60), axes="zyx")
    img[15:45, 15:45, 15:45] = 1
    img0 = img.crop_center(0.5, dims="yx")
    assert img0.shape == (60, 30, 30)
    assert img0.axes == ["z", "y", "x"]
    
    img0 = img.crop_center(0.5, dims="zyx")
    assert img0.shape == (30, 30, 30)
    assert img0.axes == ["z", "y", "x"]
    assert img0.mean() == 1.0

    img0 = img.crop_center([0.5, 1, 0.2], dims="zyx")
    assert img0.shape == (30, 60, 12)
    assert img0.axes == ["z", "y", "x"]

def test_crop_center_odd():
    img = ip.zeros((17, 17), axes="yx")
    img[4:13, 4:13] = 1
    img0 = img.crop_center(0.5)
    assert img0.shape == (9, 9)
    assert img0.mean() == 1