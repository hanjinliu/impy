import pytest
import numpy as np
import impy as ip

@pytest.mark.parametrize(
    "dtype",
    [
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
        np.float16, np.float32, np.float64,
    ]
)
def test_as_img_type(dtype):
    img = ip.zeros((4, 5, 6), dtype=dtype, axes="zyx")
    assert img.dtype == dtype
    img = ip.zeros((4, 5, 6), dtype=np.uint8, axes="zyx")
    assert img.as_img_type(dtype).dtype == dtype
