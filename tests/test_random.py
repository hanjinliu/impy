import numpy as np
from numpy.testing import assert_allclose
import impy as ip
import pytest

sizes = [None, 10, (10,), (10, 10), (6, 10, 10)]

rng = ip.random.default_rng(120)
np_rng = np.random.default_rng(120)

@pytest.mark.parametrize("size", sizes)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_random(size, dtype):
    assert_allclose(rng.random(size, dtype=dtype), np_rng.random(size, dtype=dtype))

@pytest.mark.parametrize("size", sizes)
def test_normal(size):
    assert_allclose(rng.normal(size=size), np_rng.normal(size=size))

def test_like_param():
    img = ip.zeros((3, 4, 5), axes="zyx")
    out = rng.normal(like=img)
    assert out.axes == ["z", "y", "x"]
    assert out.shape == (3, 4, 5)
