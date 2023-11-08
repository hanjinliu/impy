from __future__ import annotations

from typing import Callable, Iterable, NamedTuple, Sequence, TYPE_CHECKING
import numpy as np

from impy.axes import Axis, Axes, AxisLike
from impy.array_api import xp

if TYPE_CHECKING:
    from impy import ImgArray

__all__ = [
    "compose_affine_matrix", 
    "decompose_affine_matrix",
    "affinefit", 
    "check_matrix",
    "warp",
]

class AffineTransformationParameters(NamedTuple):
    translation: np.ndarray
    rotation: np.ndarray
    scale: np.ndarray
    shear: np.ndarray

def warp(
    img: xp.ndarray,
    matrix: xp.ndarray,
    mode="constant",
    cval=0,
    output_shape=None,
    offset=0.0,
    order=1,
    prefilter=False,
):
    img = xp.asarray(img, dtype=img.dtype)
    matrix = xp.asarray(matrix)
    out = xp.ndi.affine_transform(img, matrix, cval=cval, mode=mode, output_shape=output_shape, 
                                  offset=offset, order=order, prefilter=prefilter)
    return out

def shift(img: xp.ndarray, shift: xp.ndarray, cval=0, mode="constant", order=1, prefilter=False):
    img = xp.asarray(img, dtype=img.dtype)
    out = xp.ndi.shift(img, shift, cval=cval, mode=mode, 
                       order=order, prefilter=prefilter)
    return out

def compose_affine_matrix(
    scale=None,
    translation=None,
    rotation=None,
    shear=None,
    ndim: int = 2,
):
    from skimage.transform import AffineTransform

    # These two modules returns consistent matrix in the two dimensional case.
    # rotation must be in radian.
    if ndim == 2:
        af = AffineTransform(scale=scale, translation=translation, rotation=rotation, shear=shear)
        mx = af.params
    else:
        from napari.utils.transforms import Affine
        if scale is None:
            scale = [1] * ndim
        elif np.isscalar(scale):
            scale = [scale] * ndim
        if translation is None:
            translation = [0] * ndim
        elif np.isscalar(translation):
            translation = [translation] * ndim
        if rotation is not None:
            rotation = np.rad2deg(rotation)
        
        af = Affine(scale=scale, translate=translation, rotate=rotation, shear=shear)
        mx = af.affine_matrix
    
    return mx


def decompose_affine_matrix(matrix: np.ndarray):
    ndim = matrix.shape[0] - 1
    if ndim == 2:
        from skimage.transform import AffineTransform
        af = AffineTransform(matrix=matrix)
        out = AffineTransformationParameters(translation=af.translation, rotation=af.rotation, 
                                             scale=af.scale, shear=af.shear)
    else:
        from napari.utils.transforms import Affine
        af = Affine(affine_matrix=matrix)
        out = AffineTransformationParameters(translation=af.translate, rotation=af.rotate, 
                                             scale=af.scale, shear=af.shear)
    return out


def calc_corr(img0: np.ndarray, img1: np.ndarray, matrix, corr_func):
    """
    Calculate value of corr_func(img0, matrix(img1)).
    """
    img1_transformed = warp(img1, matrix)
    return corr_func(img0, img1_transformed)

def affinefit(img, imgref, bins=256, order=1):
    from scipy.stats import entropy
    from scipy.optimize import minimize
    from skimage.transform import warp
    
    as_3x3_matrix = lambda mtx: np.vstack((mtx.reshape(2,3), [0., 0., 1.]))
    def normalized_mutual_information(img, imgref):
        """
        Y(A,B) = (H(A)+H(B))/H(A,B)
        See "Elegant SciPy"
        """
        hist, edges = np.histogramdd([np.ravel(img), np.ravel(imgref)], bins=bins)
        hist /= np.sum(hist)
        e1 = entropy(np.sum(hist, axis=0)) # Shannon entropy
        e2 = entropy(np.sum(hist, axis=1))
        e12 = entropy(np.ravel(hist)) # mutual entropy
        return (e1 + e2)/e12
    
    def cost_nmi(mtx, img, imgref):
        mtx = as_3x3_matrix(mtx)
        img_transformed = warp(img, mtx, order=order)
        return -normalized_mutual_information(img_transformed, imgref)
    
    mtx0 = np.array([[1., 0., 0.],
                     [0., 1., 0.]]) # aberration makes little difference
    
    result = minimize(
        cost_nmi, mtx0, args=(np.asarray(img), np.asarray(imgref)), method="Powell"
    )
    mtx_opt = as_3x3_matrix(result.x)
    return mtx_opt


def check_matrix(matrices: list[np.ndarray | float]):
    """Check Affine transformation matrix."""    
    mtx = []
    for m in matrices:
        if np.isscalar(m): 
            if m == 1:
                mtx.append(m)
            else:
                raise ValueError(f"Only `1` is ok, but got {m}")
            
        elif m.shape != (3, 3) or not np.allclose(m[2,:2], 0):
            raise ValueError(f"Wrong Affine transformation matrix:\n{m}")
        
        else:
            mtx.append(m)
    return mtx

def polar2d(
    img: xp.ndarray,
    rmax: int,
    dtheta: float = 0.1,  # radian
    order: int = 1,
    mode: str = "constant",
    cval: float = 0,
) -> np.ndarray:
    centers = np.array(img.shape)/2 - 0.5
    r = xp.arange(rmax) + 0.5
    theta = xp.arange(0, 2*np.pi, dtheta)
    r, theta = xp.meshgrid(r, theta)
    y = r * xp.sin(theta) + centers[0]
    x = r * xp.cos(theta) + centers[1]
    coords = xp.stack([y, x], axis=0)
    img = xp.asarray(img, dtype=img.dtype)
    out = xp.ndi.map_coordinates(img, coords, order=order, mode=mode, cval=cval, prefilter=order>1)
    return out

def normalize_radon_input(
    self: ImgArray, 
    dims: Axes, 
    central_axis: AxisLike, 
    degrees: Sequence[float] | float,
):    
    ndim = len(dims)
    squeeze = not hasattr(degrees, "__iter__")
    
    if squeeze:
        degrees = [degrees]
    if ndim != self.ndim:
        raise NotImplementedError(
            f"Image axes is {self.axes}. Batch Radon transformation is not implemented yet. "
            "If you want to run 3D Radon transformation, convert the axes to 'zyx'."
        )
    radians = np.deg2rad(list(degrees))

    if ndim == 2:
        if central_axis is not None:
            import warnings
            warnings.warn(
                "For 2D image, the central_axis of rotation is pre-defined. "
                "This parameter will be ignored", UserWarning,
            )
        iy, ix = self.shape
        height = int(np.ceil(np.sqrt(iy ** 2 + ix ** 2)))
        output_shape = (height, self.shape[1])
        params = _get_rotation_matrices_for_radon_2d(radians, self.shape, output_shape)

    elif ndim == 3:
        # normalize central axis to a 3D vector
        if central_axis is None:
            raise ValueError("For 3D image, the central_axis of rotation must be specified.")
        elif isinstance(central_axis, (str, Axis)):
            idx = dims.index(central_axis)
            central_axis = np.zeros(ndim, dtype=np.float32)
            central_axis[idx] = 1.0
        else:
            central_axis = np.asarray(central_axis)
            central_axis /= np.sqrt(np.sum(central_axis ** 2))  # normalize
            central_axis: np.ndarray
            if central_axis.shape != (ndim,):
                raise ValueError(f"Image is {ndim}D but central_axis is {central_axis.ndim}D.")
        
        # construct Affine transform matrices
        height = int(np.ceil(np.linalg.norm(self.shape)))
        output_shape = (height, self.shape[1], self.shape[2])
        params = _get_rotation_matrices_for_radon_3d(
            radians, central_axis, self.shape, output_shape
        )
    else:
        raise ValueError("Only 2D or 3D input is supported.")
    
    return params, output_shape, squeeze

def normalize_iradon_input(
    self: ImgArray, 
    central_axis: AxisLike,
    height_axis: AxisLike,
    degree_axis: AxisLike,
    height: int,
):
    is_3d = self.ndim == 3
    if height is None:
        height = self.shape[-1]
    if height_axis is None:
        height_axis = "z" if is_3d else "y"
    if degree_axis is None:
        degree_axis = self.axes[0]
    if is_3d and central_axis not in self.axes:
        raise ValueError(
            f"central_axis={central_axis!r} does not exist in the input image. "
            f"Input has {self.axes}."
        )
    new_axes = self.axes.drop([degree_axis]).insert(0, height_axis)
    new_axes[0].scale = new_axes[-1].scale
    new_axes[0].unit = new_axes[-1].unit
    output_shape = (height, self.shape[-1])
    return central_axis, degree_axis, output_shape, new_axes

def radon_single(img: xp.ndarray, mtx: np.ndarray, order: int = 3, output_shape=None):
    """Radon transform of 2D image."""
    img_rot = warp(img, mtx, order=order, output_shape=output_shape, prefilter=False)
    return xp.sum(img_rot, axis=0)

def _get_rotation_matrices_for_radon_2d(
    radians: Sequence[float],
    in_shape: tuple[int, int],
    out_shape: tuple[int, int],
) -> Iterable[np.ndarray]:
    rotation = np.zeros((len(radians), 3, 3))
    c, s = np.cos(radians), np.sin(radians)
    rotation[:, 0, 0] = rotation[:, 1, 1] = c
    rotation[:, 0, 1] = s
    rotation[:, 1, 0] = -s
    rotation[:, 2, 2] = 1.0
    in_center = (np.array(in_shape) - 1.) / 2.
    out_center = (np.array(out_shape) - 1.) / 2.
    tr_0 = compose_affine_matrix(translation=in_center, ndim=2)
    tr_1 = compose_affine_matrix(translation=-out_center, ndim=2)
    return np.einsum("ij,njk,kl->nil", tr_0, rotation, tr_1)

def _get_rotation_matrices_for_radon_3d(
    radians: Sequence[float],
    central_axis: np.ndarray,
    in_shape: tuple[int, int, int],
    out_shape: tuple[int, int, int],
) -> Iterable[np.ndarray]:
    from scipy.spatial.transform import Rotation
    vec = np.stack([central_axis * rad for rad in radians], axis=0)
    rotation = np.zeros((len(vec), 4, 4))
    rotation[:, :3, :3] = Rotation.from_rotvec(vec).as_matrix()  # (N, 3, 3)
    rotation[:, 3, 3] = 1.0
    in_center = (np.array(in_shape) - 1.) / 2.
    out_center = (np.array(out_shape) - 1.) / 2.
    tr_0 = compose_affine_matrix(translation=in_center, ndim=3)
    tr_1 = compose_affine_matrix(translation=-out_center, ndim=3)
    return np.einsum("ij,njk,kl->nil", tr_0, rotation, tr_1)

# This function is mostly ported from `skimage.transform`.
# The most important difference is that this implementation support arbitrary 
# output shape.
def iradon(
    img: xp.ndarray, 
    degrees: xp.ndarray,
    output_shape: tuple[int, int],
    filter_func: xp.ndarray,
    interp: Callable[[np.ndarray, np.ndarray], np.ndarray],
):
    angles_count = len(degrees)
    dtype = img.dtype
    img_shape = img.shape[0]

    # Apply filter in Fourier domain
    projection = xp.fft.fft(xp.asarray(img), axis=0) * filter_func
    radon_filtered = xp.real(xp.fft.ifft(projection, axis=0)[:img_shape, :])
    radon_filtered = xp.asnumpy(radon_filtered)

    # Reconstruct image by interpolation
    reconstructed = np.zeros(output_shape, dtype=dtype)
    xpr, ypr = np.indices(output_shape)
    xpr = xpr - output_shape[0] // 2
    ypr = ypr - output_shape[1] // 2

    x = np.arange(img_shape) - img_shape // 2  # NOTE: use CPU!
    for col, angle in zip(radon_filtered.T, np.deg2rad(degrees)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        interpolant = interp(x, col)
        reconstructed += np.asarray(interpolant(t))

    return reconstructed * np.pi / (2 * angles_count)

# This function is almost ported from `skimage.transform`, but create a ramp 
# filter in a simpler way.
def get_fourier_filter(size: int, filter_name: str):
    # ramp filter
    fourier_filter = xp.zeros(size)
    split = (size + 1) // 2
    fourier_filter[:split] = np.linspace(0, 1, split)
    fourier_filter[split:] = np.linspace(1, 0, size - split)
    if filter_name == "ramp":
        pass
    elif filter_name == "shepp-logan":
        # Start from first element to avoid divide by zero
        omega = np.pi * xp.fft.fftfreq(size)[1:]
        fourier_filter[1:] *= xp.sin(omega) / omega
    elif filter_name == "cosine":
        freq = xp.linspace(0, np.pi, size, endpoint=False)
        cosine_filter = xp.fft.fftshift(xp.sin(freq))
        fourier_filter *= cosine_filter
    elif filter_name == "hamming":
        fourier_filter *= xp.fft.fftshift(xp.hamming(size))
    elif filter_name == "hann":
        fourier_filter *= xp.fft.fftshift(xp.hanning(size))
    elif filter_name is None:
        fourier_filter[:] = 1
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    return fourier_filter[:, np.newaxis]
