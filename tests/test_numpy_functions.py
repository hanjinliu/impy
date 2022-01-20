import impy as ip
import numpy as np
from numpy.testing import assert_allclose

def test_numpy_function():
    np.random.seed(1234)
    img = ip.random.random_uint16((10, 100, 100))
    assert img.axes == "tyx"
    assert isinstance(np.mean(img), float)
    proj = np.mean(img, axis="y")
    assert_allclose(proj, np.mean(img.value, axis=1))
    np.random.seed()
    