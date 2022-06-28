import numpy as np
from numpy.testing import assert_allclose
import impy as ip
import pytest

sizes = [None, 10, (10,), (10, 10), (6, 10, 10)]

rng = ip.random.default_rng(120)
np_rng = np.random.default_rng(120)

@pytest.mark.parametrize("size", sizes)
def test_random(size):
    assert_allclose(rng.random(size), np_rng.random(size))
    
@pytest.mark.parametrize("size", sizes)
def test_normal(size):
    assert_allclose(rng.normal(size=size), np_rng.normal(size=size))
