import impy as ip
from skimage.registration import phase_cross_correlation
import numpy as np
from numpy.testing import assert_allclose

def test_pcc():
    reference_image = ip.sample_image("camera")
    shift = (-7, 12)
    shifted_image = reference_image.affine(translation=shift)

    shift_sk, _, _ = phase_cross_correlation(shifted_image, reference_image)
    shift_ip = ip.pcc_maximum(shifted_image, reference_image)
    assert_allclose(shift_sk, shift_ip)
    assert_allclose(shift_sk, (7, -12))


def test_fourier():
    reference_image = ip.sample_image("camera")
    shift = (-7, 12)
    shifted_image = reference_image.affine(translation=shift)
    
    reference_image_ft = reference_image.fft()
    shifted_image_ft = shifted_image.fft()
    
    shift_sk, _, _ = phase_cross_correlation(shifted_image_ft, reference_image_ft, space="fourier")
    shift_ip = ip.ft_pcc_maximum(shifted_image_ft, reference_image_ft)
    assert_allclose(shift_sk, shift_ip)
    assert_allclose(shift_sk, (7, -12))

def test_max_shift():
    for i in range(10):
        np.random.seed(i)
        ref = ip.random.random_uint16((128, 129))
        img = ref.affine(translation=[30, -44])
        shift = ip.pcc_maximum(img, ref, max_shifts=20)
        assert all(shift < 20)