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
    # check shifts don't exceed max_shifts
    for i in range(10):
        np.random.seed(i)
        ref = ip.random.random_uint16((128, 129))
        img = ref.affine(translation=[30, -44])
        shift = ip.pcc_maximum(img, ref, max_shifts=20)
        assert all(shift <= 20)
        shift = ip.pcc_maximum(img, ref, max_shifts=14.6)
        assert all(shift <= 14.6)
    
    # check shifts are correct if max_shifts is large enough
    reference_image = ip.sample_image("camera")
    shift = (-7, 12)
    shifted_image = reference_image.affine(translation=shift)

    shift = ip.pcc_maximum(shifted_image, reference_image, max_shifts=15.7)
    assert_allclose(shift, shift)
    
    # check shifts are correct even if at the edge of max_shifts
    reference_image = ip.sample_image("camera")
    shift = (-7.8, 6.6)
    shifted_image = reference_image.affine(translation=shift)

    shift = ip.pcc_maximum(shifted_image, reference_image, max_shifts=[7.9, 6.7])
    assert_allclose(shift, shift)
    
    # check sub-optimal shifts will be returned
    ref = ip.zeros((128, 128))
    ref[10, 10] = 1
    ref[10, 20] = 1
    ref0 = ip.zeros((128, 128))
    ref0[10, 20] = 1
    img = ip.zeros((128, 128))
    img[12, 25] = 1
    img[12, 35] = 1
    
    shift = ip.pcc_maximum(img, ref)
    assert_allclose(shift, (2, 15))
    shift0 = ip.pcc_maximum(img, ref0)
    shift = ip.pcc_maximum(img, ref, max_shifts=[5.7, 10], upsample_factor=2)
    assert_allclose(shift, shift0)
    
def test_polar_pcc():
    reference_image = ip.sample_image("camera")
    deg = 21
    rotated_image = reference_image.rotate(deg)

    rot = ip.polar_pcc_maximum(rotated_image, reference_image)
    assert rot == deg

def test_fsc():
    img0 = ip.random.random_uint16((80, 80))
    img1 = ip.random.random_uint16((80, 80))
    a, b = ip.fsc(img0, img1)
    assert a.size == b.size