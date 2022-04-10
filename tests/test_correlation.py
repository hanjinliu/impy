import impy as ip
from skimage.registration import phase_cross_correlation
import numpy as np
from numpy.testing import assert_allclose

def test_pcc(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        reference_image = ip.sample_image("camera")
        shift = (-7, 12)
        shifted_image = reference_image.affine(translation=shift)

        shift_sk, _, _ = phase_cross_correlation(shifted_image, reference_image)
        shift_ip = ip.pcc_maximum(shifted_image, reference_image)
        assert_allclose(shift_sk, shift_ip)
        assert_allclose(shift_sk, (7, -12))

def test_zncc_shift(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        reference_image = ip.sample_image("camera")
        shift = (-7, 12)
        shifted_image = reference_image.affine(translation=shift)

        imgs = (shifted_image, reference_image)
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=1), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=10), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=1, max_shifts=25), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=10, max_shifts=25), (7, -12))
        
        # test different size
        imgs = (shifted_image, reference_image[20:-20, 14:-14])
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=1), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=10), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=1, max_shifts=25), (7, -12))
        assert_allclose(ip.zncc_maximum(*imgs, upsample_factor=10, max_shifts=25), (7, -12))
        

def test_cc(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        img0 = ip.random.random((10, 10, 10), axes="zyx")
        img1 = ip.random.random((10, 10, 10), axes="zyx")
        ip.ncc(img0, img1)
        ip.zncc(img0, img1)
        ip.fourier_ncc(img0, img1)
        ip.fourier_zncc(img0, img1)
        ip.ncc(img0, img1, dims="yx")
        ip.zncc(img0, img1, dims="yx")
        ip.fourier_ncc(img0, img1, dims="yx")
        ip.fourier_zncc(img0, img1, dims="yx")
        
        mask = ip.circular_mask(2, shape=img0.shape)
        ip.ncc(img0, img1, mask)
        ip.zncc(img0, img1, mask)
        ip.fourier_ncc(img0, img1)
        ip.fourier_zncc(img0, img1)
        
        mask = ip.circular_mask(2, shape=img0.shape[1:])
        ip.ncc(img0, img1, mask, dims="yx")
        ip.zncc(img0, img1, mask, dims="yx")
        ip.fourier_ncc(img0, img1, dims="yx")
        ip.fourier_zncc(img0, img1, dims="yx")
        
        assert abs(ip.zncc(img0, img0) - 1) < 1e-6
        assert abs(ip.fourier_zncc(img0, img0) - 1) < 1e-6
        

def test_fourier(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        reference_image = ip.sample_image("camera")
        shift = (-7, 12)
        shifted_image = reference_image.affine(translation=shift)
        
        reference_image_ft = reference_image.fft()
        shifted_image_ft = shifted_image.fft()
        
        shift_sk, _, _ = phase_cross_correlation(shifted_image_ft, reference_image_ft, space="fourier")
        shift_ip = ip.ft_pcc_maximum(shifted_image_ft, reference_image_ft)
        assert_allclose(shift_sk, shift_ip)
        assert_allclose(shift_sk, (7, -12))

def test_pcc_max_shift(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
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
        img = ip.zeros((128, 128))
        img[12, 25] = 1
        img[12, 35] = 1
        
        shift = ip.pcc_maximum(img, ref)
        assert_allclose(shift, (2, 15))
        shift = ip.pcc_maximum(img, ref, max_shifts=[5.7, 10], upsample_factor=2)
        assert_allclose(shift, [2, 5])


def test_zncc_max_shift(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        # check shifts don't exceed max_shifts
        for i in range(10):
            np.random.seed(i)
            ref = ip.random.random_uint16((128, 129))
            img = ref.affine(translation=[30, -44])
            shift = ip.zncc_maximum(img, ref, max_shifts=20)
            assert all(shift <= 20)
            shift = ip.zncc_maximum(img, ref, max_shifts=14.6)
            assert all(shift <= 14.6)
        
        # check shifts are correct if max_shifts is large enough
        reference_image = ip.sample_image("camera")
        shift = (-7, 12)
        shifted_image = reference_image.affine(translation=shift)

        shift = ip.zncc_maximum(shifted_image, reference_image, max_shifts=15.7)
        assert_allclose(shift, shift)
        
        # check shifts are correct even if at the edge of max_shifts
        reference_image = ip.sample_image("camera")
        shift = (-7.8, 6.6)
        shifted_image = reference_image.affine(translation=shift)

        shift = ip.zncc_maximum(shifted_image, reference_image, max_shifts=[7.9, 6.7])
        assert_allclose(shift, shift)
        
        # check sub-optimal shifts will be returned
        ref = ip.zeros((128, 128))
        ref[10, 10] = 1
        ref[10, 20] = 1
        img = ip.zeros((128, 128))
        img[12, 25] = 1
        img[12, 35] = 1
        
        shift = ip.zncc_maximum(img, ref)
        assert_allclose(shift, (2, 15))
        shift = ip.zncc_maximum(img, ref, max_shifts=[5.7, 10], upsample_factor=2)
        assert_allclose(shift, [2, 5])

def test_polar_pcc(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        reference_image = ip.sample_image("camera")
        deg = 21
        rotated_image = reference_image.rotate(deg)

        rot = ip.polar_pcc_maximum(rotated_image, reference_image)
        assert rot == deg

def test_fsc(resource):
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        img0 = ip.random.random_uint16((80, 80))
        img1 = ip.random.random_uint16((80, 80))
        a, b = ip.fsc(img0, img1)
        assert a.size == b.size

def test_landscale(resource):
    img0 = ip.sample_image("camera")[100:180, 100:180]
    shift = (-2, 3)
    img1 = img0.affine(translation=shift)
    with ip.SetConst(RESOURCE=resource, SHOW_PROGRESS=False):
        lds = ip.pcc_landscape(img0, img1, 5)
        assert lds.shape == (11, 11)
        assert np.unravel_index(np.argmax(lds), lds.shape) == (3, 8)
        lds = ip.zncc_landscape(img0, img1, 5)
        assert lds.shape == (11, 11)
        assert np.unravel_index(np.argmax(lds), lds.shape) == (3, 8)
        