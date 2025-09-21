from __future__ import annotations
import warnings
from functools import partial
from itertools import product
from typing import TYPE_CHECKING, Literal, Sequence, Iterable, Callable, overload, Any

import numpy as np
from numpy.typing import DTypeLike
from scipy import ndimage as ndi

from impy.arrays.labeledarray import LabeledArray
from impy.arrays.label import Label
from impy.arrays.phasearray import PhaseArray
from impy.arrays.specials import PropArray
from impy.arrays.tiled import TiledAccessor

from impy.arrays._utils import _filters, _linalg, _deconv, _misc, _glcm, _docs, _transform, _structures, _corr

from impy.arrays.bases.metaarray import _NoValue
from impy.utils.axesop import add_axes, switch_slice, complement_axes, find_first_appeared
from impy.utils.deco import check_input_and_output, dims_to_spatial_axes, same_dtype
from impy.utils.gauss import GaussianBackground, GaussianParticle
from impy.utils.misc import check_nd, largest_zeros
from impy.utils.slicer import solve_slicer

from impy.collections import DataDict
from impy.axes import AxisLike, slicer, Axes, Axis
from impy._types import nDInt, nDFloat, Dims, Coords, AxesTargetedSlicer, PaddingMode
from impy._const import Const
from impy.array_api import xp, cupy_dispatcher

if TYPE_CHECKING:
    from ..frame import MarkerFrame
    from typing import Literal
    ThreasholdMethod = Literal[
        "isodata", "li", "local", "mean", "min", "minimum", "niblack", "otsu", "sauvola",
        "triangle", "yen"
    ]
    FftShape = Literal["same", "square"]

class ImgArray(LabeledArray):
    """
    An n-D array for image analysis.

    Attributes
    ----------
    axes : str
        Image axes, such as "zyx" or "tcyx".
    scale : ScaleDict
        Physical scale along each axis. For instance, scale of x-axis can be referred
        to by ``img.scale["x"]`` or ``img.scale.x
    metadata : dict
        Metadata tagged to the image.
    source : Path
        Source file of the image.
    """
    tiled: TiledAccessor[ImgArray] = TiledAccessor()

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def affine(
        self, matrix=None, *, scale=None, rotation=None, shear=None, translation=None,
        order: int = 1, mode: PaddingMode = "constant", cval: float = 0, output_shape = None,
        prefilter: bool | None = None, dims: Dims = None, update: bool = False,
    ) -> ImgArray:
        r"""
        Convert image by Affine transformation. 2D Affine transformation is written as:

        .. math::

            \begin{bmatrix} y'\\ x' \\1 \end{bmatrix} =
            \begin{bmatrix} A_{00} & A_{01} & A_{02} \\
                            A_{10} & A_{11} & A_{12} \\
                                0 &      0 &      1  \end{bmatrix}
                            \begin{bmatrix} y \\x\\ 1 \end{bmatrix}


        and similarly, n-D Affine transformation can be described as (n+1)-D matrix.

        Parameters
        ----------
        matrix, scale, rotation, shear, translation
            Affine transformation parameters. See ``skimage.transform.AffineTransform`` for details.
        {order}{mode}{cval}
        output_shape : tuple of int, optional
            Shape of output array.
        {dims}{update}

        Returns
        -------
        ImgArray
            Transformed image.
        """
        if update and output_shape is not None:
            raise ValueError("Cannot update image when output_shape is provided.")
        if isinstance(cval, str) and hasattr(np, cval):
            cval = getattr(np, cval)(self.value)
        elif callable(cval):
            cval = cval(self.value)

        prefilter = prefilter or order > 1

        if translation is not None and all(a is None for a in [matrix, scale, rotation, shear]):
            shift = -np.asarray(translation)
            return self._apply_dask(
                _transform.shift,
                c_axes=complement_axes(dims, self.axes),
                kwargs=dict(shift=shift, order=order, mode=mode, cval=cval, prefilter=prefilter)
            )
        if matrix is None:
            matrix = _transform.compose_affine_matrix(
                scale=scale, rotation=rotation, shear=shear, translation=translation,
                ndim=len(dims)
            )
        return self._apply_dask(
            _transform.warp,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(matrix=matrix, order=order, mode=mode, cval=cval,
                        output_shape=output_shape, prefilter=prefilter)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def rotate(
        self,
        degree: float,
        center: Sequence[float] | Literal["center"] = "center",
        *,
        order: int = 3,
        mode: PaddingMode = "constant",
        cval: float = 0,
        dims: Dims = 2,
        update: bool = False
    ) -> ImgArray:
        """
        2D rotation of an image around a point. Outside will be padded with zero. For n-D images,
        this implementation is faster than ``scipy.ndimage.rotate``.

        Parameters
        ----------
        degree : float
            Clockwise degree of rotation. Not radian.
        center : str or array-like, optional
            Rotation center coordinate. By default the center of image will be the rotation center.
        {order}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Rotated image.
        """
        if center == "center":
            center = np.array(self.sizesof(dims))/2. - 0.5
        else:
            center = np.asarray(center)

        translation_0 = _transform.compose_affine_matrix(translation=center)
        rotation = _transform.compose_affine_matrix(rotation=np.deg2rad(degree))
        translation_1 = _transform.compose_affine_matrix(translation=-center)

        mx = translation_0 @ rotation @ translation_1
        mx[-1, :] = [0] * len(dims) + [1]
        return self._apply_dask(
            _transform.warp,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(matrix=mx, order=order, mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def stretch(
        self,
        scale: nDFloat,
        center: Sequence[float] | Literal["center"] = "center",
        *,
        mode: PaddingMode = "constant",
        cval: float = 0,
        dims: Dims = None,
        order: int = 1,
    ) -> ImgArray:
        """
        2D stretching of an image from a point.

        Parameters
        ----------
        scale: array-like
            Stretch factors.
        center : str or array-like, optional
            Rotation center coordinate. By default the center of image will be the rotation center.
        {mode}{cval}{dims}{order}

        Returns
        -------
        ImgArray
            Stretched image.
        """
        if center == "center":
            center = np.array(self.sizesof(dims))/2. - 0.5
        else:
            center = np.asarray(center)

        scale = check_nd(scale, len(dims))

        translation_0 = _transform.compose_affine_matrix(translation=center)
        stretch = _transform.compose_affine_matrix(scale=1/np.asarray(scale))
        translation_1 = _transform.compose_affine_matrix(translation=-center)

        mx = translation_0 @ stretch @ translation_1
        mx[-1, :] = [0] * len(dims) + [1]

        return self._apply_dask(
            _transform.warp,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(matrix=mx, order=order, mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def shift(
        self, translation, order: int = 3, mode: PaddingMode = "constant",
        cval: float = 0, prefilter: bool | None = None, dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        2D stretching of an image from a point.

        Parameters
        ----------
        translation : array-like
            Translation in pixels. Must match the length of spatial dimensions.
        {mode}{cval}{dims}{order}

        Returns
        -------
        ImgArray
            Stretched image.
        """
        if isinstance(cval, str) and hasattr(np, cval):
            cval = getattr(np, cval)(self.value)
        elif callable(cval):
            cval = cval(self.value)

        prefilter = prefilter or order > 1

        return self._apply_dask(
            _transform.shift,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(shift=translation, order=order, mode=mode, cval=cval,
                        prefilter=prefilter)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    def zoom(
        self,
        zoom: nDFloat,
        *,
        order: int = 3,
        mode: PaddingMode = "constant",
        cval: float = 0.0,
        same_shape: bool = False,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Zoom image by applying the diagonal component of Affine matrix.

        Parameters
        ----------
        scale : float
            Relative scale of the new image. scale 1/2 means that the shape of
            the output image will be (N/2, ...).
        {order}{mode}{cval}{dims}
        same_shape : bool, default is False
            If True, the output image will have the same shape as the input image.

        Returns
        -------
        ImgArray
            Rescaled image.
        """
        zoom_is_seq = hasattr(zoom, "__iter__")

        # Check if output is too large.
        gb = -1
        if not zoom_is_seq and zoom > 1:
            gb = np.prod(self.shape) * (zoom ** len(dims)) / 2**30 * self.itemsize
        elif zoom_is_seq and np.prod(list(zoom)) > 1:
            gb = np.prod(self.shape) * np.prod(list(zoom)) / 2**30 * self.itemsize
        if gb > Const["MAX_GB"]:
            raise MemoryError(f"Output image is too large: {gb} GB")

        if zoom_is_seq:
            zoom = list(zoom)
            if len(zoom) != len(dims):
                raise ValueError(
                    "scale must have the same length as the spatial dimensions."
                )
            it = iter(zoom)
            zoom_ = [next(it) if a in dims else 1 for a in self.axes]
        else:
            zoom_ = [zoom if a in dims else 1 for a in self.axes]

        if not same_shape:
            if mode == "constant":
                mode = "grid-constant"
            out = ndi.zoom(
                self.value, zoom_, order=order, mode=mode, cval=cval,
                prefilter=order > 1, grid_mode=True,
            ).view(self.__class__)._set_info(self)
        else:
            center = [(s - 1) / 2 for s in self.sizesof(dims)]
            mesh = np.meshgrid(
                *[
                    (np.arange(_s) - _c) / _z + _c
                    for _s, _z, _c in zip(self.sizesof(dims), zoom_, center)
                ],
                indexing="ij",
            )
            out = self.map_coordinates(mesh, order=order, mode=mode, cval=cval, dims=dims)
        out.axes = self.axes.copy()
        out.set_scale({a: self.scale[a]/scale for a, scale in zip(self.axes, zoom_)})
        return out

    def log(self, eps: float = 1e-8) -> ImgArray:
        """Calculate natural logarithm of the image, with a small epsilon."""
        return np.log(self.as_float() + eps)

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype
    def binning(
        self,
        binsize: int = 2,
        method: str | Callable[[np.ndarray], Any] = "mean",
        *,
        check_edges: bool = True,
        dims: Dims = None
    ) -> ImgArray:
        r"""
        Binning of images. This function is similar to ``rescale`` but is strictly
        binned by :math:`N \times N` blocks. Also, any numpy functions that accept
        "axis" argument are supported for reduce functions.

        Parameters
        ----------
        binsize : int, default is 2
            Bin size, such as 2x2.
        method : str or callable, default is numpy.mean
            Reduce function applied to each bin.
        check_edges : bool, default is True
            If True, only divisible ``binsize`` is accepted. If False, image is
            cropped at the end to match `binsize`.
        {dims}

        Returns
        -------
        ImgArray
            Binned image
        """
        if isinstance(method, str):
            binfunc = getattr(np, method)
        elif callable(method):
            binfunc = method
        else:
            raise TypeError("`method` must be a numpy function or callable object.")

        if binsize == 1:
            return self
        img_to_reshape, shape, scale_ = _misc.adjust_bin(
            self.value, binsize, check_edges, dims, self.axes
        )

        reshaped_img = img_to_reshape.reshape(shape)
        axes_to_reduce = tuple(i * 2 + 1 for i in range(self.ndim))
        out: np.ndarray = binfunc(reshaped_img, axis=axes_to_reduce)
        out: ImgArray = out.view(self.__class__)
        out._set_info(self)
        out.axes = self.axes.copy()  # _set_info does not pass copy so new axes must be defined here.
        out.set_scale(
            {a: self.scale[a]/scale for a, scale in zip(self.axes, scale_)}
        )
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype
    def fourier_crop(
        self,
        binsize: float = 2,
        dims: Dims = None,
    ) -> ImgArray:
        """Crop image in Fourier space.

        Parameters
        ----------
        binsize : float, default is 2
            Crop factor. If 2, the output image will have half the shape of the input
            image.
        {dims}

        Returns
        -------
        ImgArray
            Cropped image.
        """
        if binsize == 1:
            return self
        if binsize < 1:
            raise ValueError("`binsize` must be 1 or larger.")
        new_shape = list(self.shape)
        new_scale = dict(self.scale)
        for i, a in enumerate(self.axes):
            if a in dims:
                new_shape[i] = int(self.shape[i] / binsize)
                new_scale[a] = self.scale[a] * binsize
        spatial_axes = [self.axisof(a) for a in dims]
        input = xp.asarray(self.value)

        out = xp.fft.irfftn(
            _misc.fft_crop(xp.fft.rfftn(input, axes=spatial_axes), new_shape),
            s=new_shape,
            axes=spatial_axes,
        )
        out = xp.asnumpy(out)

        out_img = out.view(self.__class__)
        out_img._set_info(self)
        out_img.axes = self.axes.copy()  # _set_info does not pass copy so new axes must be defined here.
        out_img.set_scale(new_scale)
        return out_img

    @_docs.write_docs
    @dims_to_spatial_axes
    def radial_profile(
        self,
        nbin: int = 32,
        center: Iterable[float] | None = None,
        r_max: float = None,
        *,
        method: str = "mean",
        dims: Dims = None
    ) -> PropArray:
        """
        Calculate radial profile of images. Scale along each axis will be considered,
        i.e., rather ellipsoidal profile will be calculated instead if scales are
        different between axes.

        Parameters
        ----------
        nbin : int, default is 32
            Number of bins.
        center : iterable of float, optional
            The coordinate of center of radial profile. By default, the center of image
            is used.
        r_max : float, optional
            Maximum radius to make profile. Region 0 <= r < r_max will be split into
            ``nbin`` rings (or shells). **Scale must be considered** because scales of
            each axis may vary.
        method : str, default is "mean"
            Reduce function. Basic statistics functions are supported in ``scipy.ndimage``
            but their names are not consistent with those in `numpy`. Use `numpy`'s names
            here.
        {dims}

        Returns
        -------
        PropArray
            Radial profile stored in x-axis by default. If input image has tzcyx-axes, then an array
            with tcx-axes will be returned.
        """
        func = {"mean": xp.ndi.mean,
                "sum": xp.ndi.sum_labels,
                "median": xp.ndi.median,
                "max": xp.ndi.maximum,
                "min": xp.ndi.minimum,
                "std": xp.ndi.standard_deviation,
                "var": xp.ndi.variance}[method]

        spatial_shape = self.sizesof(dims)
        inds = xp.indices(spatial_shape)

        # check center
        if center is None:
            center = [s/2 for s in spatial_shape]
        elif len(center) != len(dims):
            raise ValueError(
                f"Length of `center` must match input dimensionality '{dims}'."
            )

        r = xp.sqrt(
            sum(((x - c)*self.scale[a])**2 for x, c, a in zip(inds, center, dims))
        )
        r_lim = r.max()

        # check r_max
        if r_max is None:
            r_max = r_lim
        elif r_max > r_lim or r_max <= 0:
            raise ValueError(
                f"`r_max` must be in range of 0 < r_max <= {r_lim} with this image."
            )

        # make radially separated labels
        r_rel = r/r_max
        labels = (nbin * r_rel).astype(np.uint16) + 1
        labels[r_rel >= 1] = 0

        c_axes = complement_axes(dims, self.axes)

        out = PropArray(
            np.empty(self.sizesof(c_axes)+(int(labels.max()),)),
            dtype=np.float32,
            axes=c_axes + [dims[-1]],
            source=self.source,
            metadata=self.metadata,
            propname="radial_profile"
        )
        radial_func = partial(cupy_dispatcher(func), labels=labels, index=xp.arange(1, labels.max()+1))
        for sl, img in self.iter(c_axes, exclude=dims):
            out[sl] = xp.asnumpy(radial_func(img))
        return out

    @dims_to_spatial_axes
    def gaussfit(
        self,
        scale: float = 1/16,
        p0: Sequence[float] | None = None,
        method: str = "Powell",
        mask: np.ndarray | None = None,
        dims: Dims = "yx",
    ) -> ImgArray:
        """
        Fit the image to 2-D Gaussian background.

        Parameters
        ----------
        scale : float, default is 1/16.
            Scale of rough image (to speed up fitting).
        p0 : sequence of float, optional
            Initial parameters.
        method : str, optional
            Fitting method. See `scipy.optimize.minimize`.
        mask : np.ndarray, optional,
            If given, ignore the True region from fitting.
        {dims}

        Returns
        -------
        ImgArray
            Fit image.
        """
        ndim = len(dims)
        if self.ndim > ndim:
            out = np.empty_like(self)
            c_axes = complement_axes(dims, self.axes)
            from .bases import MetaArray
            params = MetaArray(np.empty(self.shape[:-ndim], dtype=object), axes=c_axes)
            for sl, img in self.iter(c_axes, israw=True):
                fit = img.gaussfit(scale=scale, p0=p0, method=method, dims=dims)
                params[sl[:-ndim]] = fit.metadata["GaussianParameters"]
                out[sl] = fit.value
            out.metadata = out.metadata.copy()
            out.metadata["GaussianParameters"] = params
            return out

        rough = self.zoom(scale, mode="reflect", order=1).value.astype(np.float32)
        if mask is not None:
            from impy.core import asarray as ip_asarray
            mask = ip_asarray(mask).zoom(scale, mode="reflect", order=1) > 0
        gaussian = GaussianBackground(p0)
        gaussian.fit(rough, method=method, mask=mask)
        gaussian.rescale(1/scale)
        fit = gaussian.generate(self.shape).view(self.__class__)
        fit._set_info(self)
        fit.metadata = fit.metadata.copy()
        fit.metadata["GaussianParameters"] = gaussian.asdict()
        return fit

    @same_dtype(asfloat=True)
    @check_input_and_output
    def gauss_correction(
        self,
        ref: ImgArray | None = None,
        scale: float = 1/16,
        median_radius: float = 15,
    ):
        """
        Correct unevenly distributed excitation light using Gaussian fitting.

        This method subtracts background intensity at the same time. If input image is
        uint, then output value under 0 will replaced with 0. If you want to quantify
        background, it is necessary to first convert input image to float image.

        Parameters
        ----------
        ref : ImgArray, default is `self`.
            Reference image to estimate background.
        scale : float, default is 1/16.
            Scale of rough image (to speed up fitting).
        median_radius : float, default is 15.
            Radius of median prefilter's kernel. If smaller than 1, prefiltering will be skipped.

        Returns
        -------
        ImgArray
            Corrected and background subtracted image.

        Examples
        --------
        1. When input image has "ptcyx"-axes, and you want to estimate the background intensity
        for each channel by averaging all the positions and times.

            >>> img_cor = img.gauss_correction(ref=img.proj("pt"))

        2. When input image has "ptcyx"-axes, and you want to estimate the background intensity
        for each channel and time point by averaging all the positions.

            >>> img_cor = img.gauss_correction(ref=img.proj("p"))
        """
        if ref is None:
            ref = self
        elif not isinstance(ref, self.__class__):
            raise TypeError(f"`ref` must be None or ImgArray, but got {type(ref)}")

        self_loop_axes = complement_axes(ref.axes, self.axes)
        ref_loop_axes = complement_axes("yx", ref.axes)
        out = np.empty(self.shape, dtype=np.float32)
        for sl0, ref_ in ref.iter(ref_loop_axes, israw=True):
            ref_: ImgArray
            if median_radius >= 1:
                ref_ = ref_.median_filter(radius=median_radius)
            fit = ref_.gaussfit(scale=scale)
            a = fit.max()
            for sl, img in self.iter(self_loop_axes, israw=True):
                out[sl][sl0] = (img[sl0] / fit * a - a).value

        return out.view(self.__class__)


    @_docs.write_docs
    @dims_to_spatial_axes
    def hessian_eigval(self, sigma: nDFloat = 1, *, dims: Dims = None) -> ImgArray:
        """
        Calculate Hessian's eigenvalues for each image.

        Parameters
        ----------
        {sigma}
        {dims}

        Returns
        -------
        ImgArray
            Array of eigenvalues. The axis ``"base"`` denotes the index of eigenvalues.
            l=0 means the smallest eigenvalue.

        Examples
        --------
        Extract filament
            >>> eig = -img.hessian_eigval()[ip.slicer.base[0]]
            >>> eig[eig<0] = 0
        """
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])

        eigval = self.as_float()._apply_dask(
            _linalg.hessian_eigval,
            c_axes=complement_axes(dims, self.axes),
            new_axis=-1,
            args=(sigma, pxsize)
        )

        eigval: ImgArray = np.moveaxis(eigval, -1, 0)

        new_axes = ["base"] + self.axes
        eigval._set_info(self, new_axes=new_axes)

        return eigval

    @_docs.write_docs
    @dims_to_spatial_axes
    def hessian_eig(self, sigma: nDFloat = 1, *, dims: Dims = None) -> tuple[ImgArray, ImgArray]:
        """
        Calculate Hessian's eigenvalues and eigenvectors.

        Parameters
        ----------
        {sigma}{dims}

        Returns
        -------
        ImgArray and ImgArray
            Arrays of eigenvalues and eigenvectors. The axis ``"base"`` denotes the index of
            eigenvalues. l=0 means the smallest eigenvalue. ``"dim"`` denotes the index of
            spatial dimensions. For 3D image, dim=0 means z-element of an eigenvector.
        """
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])

        eigs = self.as_float()._apply_dask(
            _linalg.hessian_eigh,
            c_axes=complement_axes(dims, self.axes),
            new_axis=[-2, -1],
            args=(sigma, pxsize)
        )

        eigval, eigvec = _linalg.eigs_post_process(eigs, self.axes, self)
        return eigval, eigvec

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def structure_tensor_eigval(self, sigma: nDFloat = 1, *, dims: Dims = None) -> ImgArray:
        """
        Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        {sigma}
        {dims}

        Returns
        -------
        ImgArray
            Array of eigenvalues. The axis ``"l"`` denotes the index of eigenvalues.
            l=0 means the smallest eigenvalue.
        """
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])

        eigval = self.as_float()._apply_dask(
            _linalg.structure_tensor_eigval,
            c_axes=complement_axes(dims, self.axes),
            new_axis=-1,
            args=(sigma, pxsize),
        )

        new_axes = ["base"] + self.axes
        eigval._set_info(self, new_axes=new_axes)

        return eigval

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def structure_tensor_eig(self, sigma: nDFloat = 1, *, dims: Dims = None)-> tuple[ImgArray, ImgArray]:
        """Calculate structure tensor's eigenvalues and eigenvectors.

        Parameters
        ----------
        {sigma}
        {dims}

        Returns
        -------
        ImgArray and ImgArray
            Arrays of eigenvalues and eigenvectors. The axis ``"l"`` denotes the index of
            eigenvalues. l=0 means the smallest eigenvalue. ``"r"`` denotes the index of
            spatial dimensions. For 3D image, r=0 means z-element of an eigenvector.
        """
        ndim = len(dims)
        sigma = check_nd(sigma, ndim)
        pxsize = np.array([self.scale[a] for a in dims])

        eigs = self.as_float()._apply_dask(
            _linalg.structure_tensor_eigh,
            c_axes=complement_axes(dims, self.axes),
            new_axis=[-2, -1],
            args=(sigma, pxsize)
        )

        eigval, eigvec = _linalg.eigs_post_process(eigs, self.axes, self)

        return eigval, eigvec

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def edge_filter(
        self,
        method: str = "sobel",
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        """Edge detection such as Sobel filter.

        Parameters
        ----------
        method : str, {"sobel", "farid", "scharr", "prewitt"}, default is "sobel"
            Edge operator name.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        from skimage.filters import sobel, farid, scharr, prewitt
        # Get operator
        method_dict = {"sobel": sobel, "farid": farid, "scharr": scharr, "prewitt": prewitt}
        try:
            f = method_dict[method]
        except KeyError:
            raise ValueError("`method` must be 'sobel', 'farid' 'scharr', or 'prewitt'.")

        return self._apply_dask(f, c_axes=complement_axes(dims, self.axes))

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def lowpass_filter(
        self,
        cutoff: nDFloat = 0.2,
        order: float = 2,
        *,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Butterworth low-pass filter.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        from ._utils._skimage import _get_ND_butterworth_filter
        ndims = len(dims)
        cutoff = check_nd(cutoff, ndims)
        if all((c >= 0.5*np.sqrt(ndims) or c <= 0) for c in cutoff):
            return self
        spatial_shape = self.sizesof(dims)
        spatial_axes = [self.axisof(a) for a in dims]
        weight = _get_ND_butterworth_filter(spatial_shape, cutoff, order, False, True)
        input = xp.asarray(self.value)
        if len(dims) < self.ndim:
            weight = add_axes(self.axes, self.shape, weight, dims)
        out = xp.fft.irfftn(
            xp.asarray(weight) * xp.fft.rfftn(input, axes=spatial_axes),
            s=spatial_shape,
            axes=spatial_axes,
        )
        return xp.asnumpy(out)

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def lowpass_conv_filter(
        self,
        cutoff: nDFloat = 0.2,
        order: float = 2,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        """Butterworth low-pass filter in real space.

        Butterworth kernel is created first using inverse Fourier transform of weight
        function.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        from ._utils._skimage import _get_ND_butterworth_filter
        ndims = len(dims)
        cutoff = check_nd(cutoff, ndims)
        if all((c >= 0.5*np.sqrt(ndims) or c <= 0) for c in cutoff):
            return self
        spatial_shape = self.sizesof(dims)
        weight = _get_ND_butterworth_filter(spatial_shape, cutoff, order, False, True)
        ker_all = xp.asnumpy(xp.fft.irfftn(xp.asarray(weight), s=spatial_shape))
        ker_all = np.fft.fftshift(ker_all)
        sl = []
        for s, c in zip(spatial_shape, cutoff):
            radius = int(min(1/c, 11))
            sl.append(slice(s//2 - radius, s//2 + radius + 1))
        ker = ker_all[tuple(sl)]
        return self.convolve(ker)

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def highpass_filter(
        self,
        cutoff: nDFloat = 0.2,
        order: float = 2,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        """Butterworth high-pass filter.

        Parameters
        ----------
        cutoff : float or array-like, default is 0.2
            Cutoff frequency.
        order : float, default is 2
            Steepness of cutoff.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        from ._utils._skimage import _get_ND_butterworth_filter
        cutoff = check_nd(cutoff, len(dims))
        spatial_shape = self.sizesof(dims)
        spatial_axes = [self.axisof(a) for a in dims]
        weight = _get_ND_butterworth_filter(spatial_shape, cutoff, order, True, True)
        input = xp.asarray(self.value)
        if len(dims) < self.ndim:
            weight = add_axes(self.axes, self.shape, weight, dims)
        out = xp.fft.irfftn(xp.asarray(weight)*xp.fft.rfftn(input, axes=spatial_axes),
                            s=spatial_shape, axes=spatial_axes)
        return xp.asnumpy(out)


    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def bandpass_filter(
        self,
        cuton: nDFloat = 0.02,
        cutoff: nDFloat = 0.2,
        order: float = 2,
        *,
        dims: Dims = None,
        update: bool = False
    ):
        """Butterworth band-pass filter.

        Parameters
        ----------
        cuton, cutoff : float or array-like
            Cuton and Cutoff frequency. Fequency domain between these two values will be
            passed.
        order : float, default is 2
            Steepness of cutoff.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        from ._utils._skimage import _get_ND_butterworth_filter
        cuton = check_nd(cuton, len(dims))
        cutoff = check_nd(cutoff, len(dims))
        spatial_shape = self.sizesof(dims)
        spatial_axes = [self.axisof(a) for a in dims]
        weight_on = _get_ND_butterworth_filter(spatial_shape, cuton, order, True, True)
        weight_off = _get_ND_butterworth_filter(spatial_shape, cutoff, order, False, True)
        weight = weight_off * weight_on
        input = xp.asarray(self.value)
        if len(dims) < self.ndim:
            weight = add_axes(self.axes, self.shape, weight, dims)
        out = xp.fft.irfftn(xp.asarray(weight)*xp.fft.rfftn(input, axes=spatial_axes),
                            s=spatial_shape, axes=spatial_axes)
        return xp.asnumpy(out)

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def convolve(
        self,
        kernel,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """General linear convolution by running kernel filtering.

        Parameters
        ----------
        kernel : array-like
            Convolution kernel.
        {mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Convolved image.
        """
        kernel = np.asarray(kernel, dtype=np.float32)
        return self._apply_dask(
            _filters.convolve,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            args=(kernel,),
            kwargs=dict(mode=mode, cval=cval)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def erosion(
        self,
        radius: float = 1.,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Morphological erosion.

        If input is binary image, the running function will automatically switched to
        ``binary_erosion`` to speed up calculation.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == bool:
            f = _filters.binary_erosion
            kwargs = dict(structure=disk, border_value=1)
        else:
            f = _filters.erosion
            kwargs = dict(footprint=disk, mode=mode, cval=cval)
        return self._apply_dask(
            f,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=kwargs
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def dilation(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Morphological dilation.

        If input is binary image, the running function will automatically switched to
        ``binary_dilation`` to speed up calculation.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == bool:
            f = _filters.binary_dilation
            kwargs = dict(structure=disk)
        else:
            f = _filters.dilation
            kwargs = dict(footprint=disk, mode=mode, cval=cval)
        return self._apply_dask(
            f,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=kwargs
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def opening(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Morphological opening.

        If input is binary image, the running function will automatically switched to
        ``binary_opening`` to speed up calculation.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == bool:
            f = _filters.binary_opening
            kwargs = dict(structure=disk)
        else:
            f = _filters.opening
            kwargs = dict(footprint=disk, mode=mode, cval=cval)
        return self._apply_dask(
            f,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=kwargs
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def closing(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Morphological closing.

        If input is binary image, the running function will automatically switched to
        ``binary_closing`` to speed up calculation.

        Parameters
        ----------
        {radius}{dims}{mode}{cval}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == bool:
            f = _filters.binary_closing
            kwargs = dict(structure=disk)
        else:
            f = _filters.closing
            kwargs = dict(footprint=disk, mode=mode, cval=cval)
        return self._apply_dask(
            f,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=kwargs
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def tophat(
        self,
        radius: float = 30,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """Tophat morphological image processing. This is useful for background subtraction.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """

        disk = _structures.ball_like(radius, len(dims))
        return self._apply_dask(
            _filters.white_tophat,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=dict(footprint=disk, mode=mode, cval=cval)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def smooth_mask(
        self,
        sigma: float = 2.0,
        dilate_radius: float = 2.0,
        mask_light: bool = False,
        *,
        dims: Dims = None
    ) -> ImgArray:
        """Smoothen binary mask image at its edges. This is useful to make a "soft mask".

        This method applies erosion/dilation to a binary image and then smooth its edges by Gaussian.
        The total value is always larger after Gaussian smoothing.

        Parameters
        ----------
        sigma : float, default is 2.0
            Standard deviation of Gaussian blur.
        dilate_radius : float, default is 2.0
            Radius in pixel that will be used to dilate mask before smoothing.
        mask_light : bool, default is False
            If true, mask array is considered to mask other image at its True values. Otherwise mask array
            plays more like a weight array, that is, False region will be zero.
        {dims}

        Returns
        -------
        ImgArray
            Smoothened mask image.
        """
        if not mask_light:
            self = ~self

        if dilate_radius > 0:
            self = self.erosion(dilate_radius, dims=dims)
        elif dilate_radius < 0:
            self = self.dilation(-dilate_radius, dims=dims)

        if sigma > 0:
            dist = self.distance_map(dims=dims)
            blurred_mask = np.exp(-dist**2/2/sigma**2)
        elif sigma == 0:
            blurred_mask = 1 - self.astype(np.float32)
        else:
            raise ValueError("sigma must be non-negative.")
        if mask_light:
            blurred_mask = 1 - blurred_mask
        return blurred_mask

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def mean_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False) -> ImgArray:
        """Mean filter. Kernel is filled with same values.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_dask(
            _filters.mean_filter,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            args=(disk,),
            kwargs=dict(mode=mode, cval=cval),
        )


    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def min_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
    ) -> ImgArray:
        """Minimum filter.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}

        Returns
        -------
        ImgArray
            Filtered image
        """
        disk = _structures.ball_like(radius, len(dims))
        return self.as_float()._apply_dask(
            _filters.min_filter,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def max_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
    ) -> ImgArray:
        """Maximum filter.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}

        Returns
        -------
        ImgArray
            Filtered image
        """
        disk = _structures.ball_like(radius, len(dims))
        return self.as_float()._apply_dask(
            _filters.max_filter,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def std_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Standard deviation filter.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}

        Returns
        -------
        ImgArray
            Filtered image
        """
        disk = _structures.ball_like(radius, len(dims))
        return self.as_float()._apply_dask(
            _filters.std_filter,
            c_axes=complement_axes(dims, self.axes),
            args=(disk,),
            kwargs=dict(mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def coef_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None
    ) -> ImgArray:
        r"""
        Coefficient of variance filter.

        For kernel area X, :math:`\frac{\sqrt{V[X]}}{E[X]}` is calculated. This filter
        is useful for feature extraction from images with uneven background intensity.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}

        Returns
        -------
        ImgArray
            Filtered image
        """
        disk = _structures.ball_like(radius, len(dims))
        return self.as_float()._apply_dask(
            _filters.coef_filter,
            c_axes=complement_axes(dims, self.axes),
            args=(disk,),
            kwargs=dict(mode=mode, cval=cval),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def median_filter(
        self,
        radius: float = 1,
        *,
        mode: PaddingMode = "reflect",
        cval: float = 0.0,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        Running multi-dimensional median filter.

        This filter is useful for deleting outliers generated by noise.

        Parameters
        ----------
        {radius}{mode}{cval}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        disk = _structures.ball_like(radius, len(dims))
        return self._apply_dask(
            _filters.median_filter,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            kwargs=dict(footprint=disk, mode=mode, cval=cval),
        )

    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output
    def diameter_opening(self, diameter:int=8, *, connectivity:int=1, dims: Dims = None,
                         update: bool = False) -> ImgArray:
        from skimage.morphology import diameter_opening
        return self._apply_dask(
            diameter_opening,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(diameter_threshold=diameter, connectivity=connectivity)
        )

    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output
    def diameter_closing(self, diameter:int=8, *, connectivity:int=1, dims: Dims = None,
                         update: bool = False) -> ImgArray:
        from skimage.morphology import diameter_closing
        return self._apply_dask(
            diameter_closing,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(diameter_threshold=diameter, connectivity=connectivity)
        )

    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output
    def area_opening(self, area:int=64, *, connectivity:int=1, dims: Dims = None,
                     update: bool = False) -> ImgArray:
        from skimage.morphology import remove_small_holes, area_opening
        if self.dtype == bool:
            return self._apply_dask(
                remove_small_holes,
                c_axes=complement_axes(dims, self.axes),
                kwargs=dict(area_threshold=area, connectivity=connectivity)
            )
        else:
            return self._apply_dask(
                area_opening,
                c_axes=complement_axes(dims, self.axes),
                kwargs=dict(area_threshold=area, connectivity=connectivity)
            )

    @dims_to_spatial_axes
    @same_dtype
    @check_input_and_output
    def area_closing(self, area:int=64, *, connectivity:int=1, dims: Dims = None,
                     update: bool = False) -> ImgArray:
        from skimage.morphology import remove_small_holes, area_closing
        if self.dtype == bool:
            return self._apply_dask(
                remove_small_holes,
                c_axes=complement_axes(dims, self.axes),
                kwargs=dict(area_threshold=area, connectivity=connectivity)
            )
        else:
            return self._apply_dask(
                area_closing,
                c_axes=complement_axes(dims, self.axes),
                kwargs=dict(area_threshold=area, connectivity=connectivity)
            )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def entropy_filter(self, radius: nDFloat = 5, *, dims: Dims = None) -> ImgArray:
        """
        Running entropy filter.

        This filter is useful for detecting change in background distribution.

        Parameters
        ----------
        {radius}{dims}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        from skimage.filters.rank import entropy
        disk = _structures.ball_like(radius, len(dims))

        self = self.as_uint8()
        return self._apply_dask(
            entropy,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(footprint=disk),
        ).as_float()

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def enhance_contrast(self, radius:nDFloat=1, *, dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Enhance contrast filter.

        Parameters
        ----------
        {radius}{dims}{update}

        Returns
        -------
        ImgArray
            Contrast enhanced image.
        """
        from skimage.filters.rank import enhance_contrast

        disk = _structures.ball_like(radius, len(dims))
        if self.dtype == np.float32:
            amp = max(np.abs(self.range))
            self.value[:] /= amp
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = self._apply_dask(
                enhance_contrast,
                c_axes=complement_axes(dims, self.axes),
                dtype=self.dtype,
                args=(disk,)
            )
        if self.dtype == np.float32:
            self.value[:] *= amp

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def laplacian_filter(self, radius: int = 1, *, dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Edge detection using Laplacian filter. Kernel is made by `skimage`'s function.

        Parameters
        ----------
        radius : int, default is 1
            Radius of kernel. Shape of kernel will be (2*radius+1, 2*radius+1).
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        from skimage.restoration.uft import laplacian
        ndim = len(dims)
        _, laplace_op = laplacian(ndim, (2 * radius + 1,) * ndim)
        return self.as_float()._apply_dask(
            _filters.convolve,
            c_axes=complement_axes(dims, self.axes),
            dtype=self.dtype,
            args=(laplace_op,),
            kwargs=dict(mode="reflect")
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def kalman_filter(self, gain: float = 0.8, noise_var: float = 0.05, *, along: str = "t",
                      dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Kalman filter for image smoothing.

        This function is same as "Kalman Stack Filter" in ImageJ but support batch
        processing. This filter is useful for preprocessing of particle tracking.

        Parameters
        ----------
        gain : float, default is 0.8
            Filter gain.
        noise_var : float, default is 0.05
            Initial estimate of noise variance.
        along : str, default is "t"
            Which axis will be the time axis.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        t_axis = self.axisof(along)
        min_a = min(self.axisof(a) for a in dims)
        if t_axis > min_a:
            self = np.swapaxes(self, t_axis, min_a)
        out = self._apply_dask(
            _filters.kalman_filter,
            c_axes=complement_axes([along] + dims, self.axes),
            args=(gain, noise_var)
        )

        if t_axis > min_a:
            out = np.swapaxes(out, min_a, t_axis)

        return out

    @dims_to_spatial_axes
    @check_input_and_output
    def focus_map(self, radius: int = 1, *, dims: Dims = 2) -> PropArray:
        """
        Compute focus map using variance of Laplacian method.

        yx-plane with higher variance is likely a focal plane because sharper image
        causes higher value of Laplacian on the edges.

        Parameters
        ----------
        radius : int, default is 1
            Radius of Laplacian filter's kernel.
        {dims}

        Returns
        -------
        PropArray
            Array of variance of Laplacian

        Examples
        --------
        Get the focus plane from a 3D image.
            >>> score = img.focus_map()
            >>> score.plot()               # plot the variation of laplacian focus
            >>> z_focus = np.argmax(score) # determine the focus plane
            >>> img[z_focus]               # get the focus plane
        """
        c_axes = complement_axes(dims, self.axes)
        laplace_img = self.as_float().laplacian_filter(radius, dims=dims)
        out = np.var(laplace_img, axis=dims)
        out = PropArray(out.value, dtype=np.float32, name=self.name,
                        axes=c_axes, propname="variance_of_laplacian")
        return out

    @_docs.write_docs
    @same_dtype(asfloat=True)
    @check_input_and_output
    def unmix(self, matrix, bg = None, *, along: str = "c", update: bool = False) -> ImgArray:
        r"""
        Unmix fluorescence leakage between channels in a linear way. For example, a blue/green image,
        fluorescent leakage can be written as following equation:

            :math:`\left\{\begin{array}{ll} B_{obs} = B_{real} + a \cdot G_{real} & \\G_{obs} = b \cdot B_{real} + G_{real} & \end{array} \right.`

        where "obs" means observed intensities, "real" means the real intensity. In this linear case,
        leakage matrix:

            :math:`M = \begin{bmatrix} 1 & a \\b & 1 \end{bmatrix} \\`

            :math:`V_{obs} = M \cdot V_{real}`

        must be predefined. If M is given, then real intensities can be restored by:

            :math:`V_{real} = M^{-1} \cdot V_{obs}`

        Parameters
        ----------
        matrix : array-like
            Leakage matrix. The (i, j) component of the matrix is the leakage from i-th channel to
            j-th channel.
        bg : array-like, optional
            Vector of background intensities for each channel. If not given, it is assumed to be the
            minimum value of each channel.
        along : str, default is "c"
            The axis of channel.
        {update}

        Returns
        -------
        ImgArray
            Unmixed image.

        Examples
        --------
        Complement the channel-0 to channel-1 leakage.
            >>> mtx = [[1.0, 0.4],
            >>>        [0.0, 1.0]]
            >>> bg = [1500, 1200]
            >>> unmixed_img = img.unmix(mtx, bg)
        """
        n_chn = self.sizeof(along)
        c_ax = self.axisof(along)

        # check matrix and bg
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.shape != (n_chn, n_chn):
            raise ValueError(f"`map_matrix` must have shape {(n_chn, n_chn)}")
        if bg is None:
            bg = np.zeros(n_chn)
        else:
            bg = np.asarray(bg, dtype=np.float32).ravel()
            if bg.size != n_chn:
                raise ValueError(f"`bg` must have length {n_chn}")

        # move channel axis to the last
        input_ = np.moveaxis(np.asarray(self, dtype=np.float32), c_ax, -1)
        # multiply inverse matrix
        out = (input_ - bg) @ np.linalg.inv(matrix) + bg
        # restore the axes order
        out = np.moveaxis(out, -1, c_ax)

        return out.view(self.__class__)


    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def fill_hole(self, thr: float|str = "otsu", *, dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Filling holes. See skimage's documents
        `here <https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_holes_and_peaks.html>`_.

        Parameters
        ----------
        thr : scalar or str, optional
            Threshold (value or method) to apply if image is not binary.
        {dims}{update}

        Returns
        -------
        ImgArray
            Hole-filled image.
        """
        if self.dtype != bool:
            mask = self.threshold(thr=thr).value
        else:
            mask = self.value

        return self._apply_dask(
            _filters.fill_hole,
            c_axes=complement_axes(dims, self.axes),
            args=(mask,),
            dtype=self.dtype
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def gaussian_filter(
        self,
        sigma: nDFloat = 1,
        *,
        fourier: bool = False,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        Run Gaussian filter (Gaussian blur).

        Parameters
        ----------
        {sigma}{fourier}{dims}{update}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        filter_func = _filters.gaussian_filter_fourier if fourier else _filters.gaussian_filter
        return self._apply_dask(
            filter_func,
            c_axes=complement_axes(dims, self.axes),
            args=(sigma,),
            dtype=np.float32
        )


    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def dog_filter(
        self,
        low_sigma: nDFloat = 1,
        high_sigma: nDFloat = None,
        *,
        fourier: bool = False,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Run Difference of Gaussian filter.

        This function does not support `update` argument because intensity can be
        negative.

        Parameters
        ----------
        low_sigma : scalar or array of scalars, default is 1.
            lower standard deviation(s) of Gaussian.
        high_sigma : scalar or array of scalars, default is x1.6 of low_sigma.
            higher standard deviation(s) of Gaussian.
        {fourier}{dims}

        Returns
        -------
        ImgArray
            Filtered image.
        """

        low_sigma = np.array(check_nd(low_sigma, len(dims)))
        high_sigma = low_sigma * 1.6 if high_sigma is None else high_sigma
        filter_func = _filters.dog_filter_fourier if fourier else _filters.dog_filter
        return self.as_float()._apply_dask(
            filter_func,
            c_axes=complement_axes(dims, self.axes),
            args=(low_sigma, high_sigma)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def doh_filter(self, sigma: nDFloat = 1, *, dims: Dims = None) -> ImgArray:
        """
        Determinant of Hessian filter. This function does not support `update`
        argument because output has total different scale of intensity.

            .. warning::
                Because in most cases we want to find only bright dots, eigenvalues larger
                than 0 is ignored before computing determinant.

        Parameters
        ----------
        {sigma}{dims}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        sigma = check_nd(sigma, len(dims))
        pxsize = np.array([self.scale[a] for a in dims])
        return self.as_float()._apply_dask(
            _filters.doh_filter,
            c_axes=complement_axes(dims, self.axes),
            args=(sigma, pxsize)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def log_filter(self, sigma: nDFloat = 1, *, dims: Dims = None) -> ImgArray:
        """
        Laplacian of Gaussian filter.

        Parameters
        ----------
        {sigma}{dims}

        Returns
        -------
        ImgArray
            Filtered image.
        """
        return -self.as_float()._apply_dask(
            _filters.gaussian_laplace,
            c_axes=complement_axes(dims, self.axes),
            args=(sigma,)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def rolling_ball(
        self,
        radius: float = 30,
        prefilter: Literal["mean", "median", "none"] = "mean",
        *,
        return_bg: bool = False,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        """
        Subtract Background using rolling-ball algorithm.

        Parameters
        ----------
        {radius}
        prefilter : "mean", "median" or "none"
            If apply 3x3 averaging before creating background.
        {dims}{update}

        Returns
        -------
        ImgArray
            Background subtracted image.
        """
        from skimage.restoration import rolling_ball
        c_axes = complement_axes(dims, self.axes)
        if prefilter == "mean":
            filt = self._apply_dask(
                _filters.mean_filter,
                c_axes=c_axes,
                kwargs=dict(selem=np.ones((3,)*len(dims)))
            )
        elif prefilter == "median":
            filt = self._apply_dask(
                _filters.median_filter,
                c_axes=c_axes,
                kwargs=dict(footprint=np.ones((3,)*len(dims)))
            )
        elif prefilter == "none":
            filt = self
        else:
            raise ValueError("`prefilter` must be 'mean', 'median' or 'none'.")
        filt.axes = self.axes
        back = filt._apply_dask(
            rolling_ball,
            c_axes=c_axes,
            kwargs=dict(radius=radius)
        )
        if not return_bg:
            out = self.value - back
            return out
        else:
            return back

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def rof_filter(
        self,
        lmd: float = 0.05,
        tol: float = 1e-4,
        max_iter: int = 50,
        *,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        Rudin-Osher-Fatemi's total variation denoising.

        Parameters
        ----------
        lmd : float, default is 0.05
            Constant value in total variation.
        tol : float, default is 1e-4
            Iteration stops when gain is under this value.
        max_iter : int, default is 50
            Maximum number of iterations.
        {dims}{update}

        Returns
        -------
        ImgArray
            Filtered image
        """
        from skimage.restoration._denoise import _denoise_tv_chambolle_nd
        return self._apply_dask(
            _denoise_tv_chambolle_nd,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(weight=lmd, eps=tol, max_num_iter=max_iter)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def wavelet_denoising(
        self,
        noise_sigma: float | None = None,
        *,
        wavelet: str = "db1",
        mode: Literal["soft", "hard"] = "soft",
        wavelet_levels: int | None = None,
        method: Literal["BayesShrink", "VisuShrink"] = "BayesShrink",
        max_shifts: int | tuple[int, ...] = 0,
        shift_steps: int | tuple[int, ...] = 1,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Wavelet denoising. Because it is not shift invariant, ``cycle_spin`` is called inside the
        function.

        Parameters
        ----------
        noise_sigma : float, optional
            Standard deviation of noise, if known.
        wavelet : str, default is "db1"
            Any options of `pywt.wavelist`.
        mode : {"soft", "hard"}, default is "soft"
            Type of denoising.
        wavelet_levels : int, optional
            The number of wavelet decomposition levels to use.
        method : {"BayesShrink", "VisuShrink"}, default is "BayesShrink"
            Thresholding method to be used
        max_shifts : int or tuple, default is 0
            Shifts in range(0, max_shifts+1) will be used.
        shift_steps : int or tuple, default is 1
            Step size of shifts.
        {dims}

        Returns
        -------
        ImgArray
            Denoised image.
        """
        from skimage.restoration import cycle_spin, denoise_wavelet
        func_kw = dict(
            sigma=noise_sigma,
            wavelet=wavelet,
            mode=mode,
            wavelet_levels=wavelet_levels,
            method=method
        )
        return self._apply_dask(
            cycle_spin,
            c_axes=complement_axes(dims, self.axes),
            args=(denoise_wavelet,),
            kwargs=dict(func_kw=func_kw, max_shifts=max_shifts, shift_steps=shift_steps)
        )

    @_docs.write_docs
    def split_pixel_unit(
        self,
        center: tuple[float, float] = (0.5, 0.5),
        *,
        order: int = 1,
        angle_order: list[int] | None = None,
        newaxis: AxisLike = "a",
    ) -> ImgArray:
        r"""
        Split a (2N, 2M)-image into four (N, M)-images for each other pixels.

        Generally, image acquisition with a polarization camera will output
        :math:`(2N, 2M)`-image with :math:`N \times M` pixel units:

        +-+-+-+-+-+-+
        |0|1|0|1|0|1|
        +-+-+-+-+-+-+
        |3|2|3|2|3|2|
        +-+-+-+-+-+-+
        |0|1|0|1|0|1|
        +-+-+-+-+-+-+
        |3|2|3|2|3|2|
        +-+-+-+-+-+-+

        This function generates images only consist of positions of [0], [1],
        [2] or [3]. Strictly, each image is acquired from different position
        (the pixel (i, j) in [0]-image and the pixel (i, j) in [1]-image are
        acquired from different positions). This function also complements for
        this difference by interpolation.

        Parameters
        ----------
        center : tuple, default is (0, 0)
            Coordinate that will be considered as the center of the returned
            image. Input (a, b) must satisfy 0 < a < 1 and 0 < b < 1. For example,
            center=(0, 0) means the most upper left pixel, and center=(0.5, 0.5)
            means the middle point of a pixel unit. `[[0, 1], [3, 2]]` becomes
            `[[(0, 0), (0, 1)], [(1, 0), (1, 1)]]`.

        {order}
        angle_order : list of int, default is [2, 1, 0, 3]
            Specify which pixels correspond to which polarization angles. 0, 1, 2
            and 3 corresponds to polarization of 0, 45, 90 and 135 degree
            respectively. This list will be directly passed to ``np.ndarray`` like
            ``arr[angle_order]`` to sort it. For example, if a pixel unit receives
            polarized light like below:

                .. code-block::

                    [0] [1]    [ 90] [ 45]    [|] [/]
                    [2] [3] -> [135] [  0] or [\] [-]

            then ``angle_order`` should be [2, 1, 0, 3].

        Returns
        -------
        ImgArray
            Axis "a" is added in the first dimension. For example, If input is
            "tyx"-axes, then output will be "atyx"-axes.

        Examples
        --------
        Extract polarization in 0-, 45-, 90- and 135-degree directions from an
        image that is acquired from a polarization camera, and calculate total
        intensity of light by averaging.

            >>> img_pol = img.split_pixel_unit()
            >>> img_total = img_pol.proj(axis="a")
        """
        yc, xc = center
        if angle_order is None:
            angle_order = [2, 1, 0, 3]

        if not self.shape.x % 2 == self.shape.y % 2 == 0:
            raise ValueError(
                f"Image pixel sizes must be even numbers, got {self.sizesof('yx')}"
            )
        imgs: list[ImgArray] = []
        fmt = slicer.get_formatter(["y", "x"])
        for y, x in [(0, 0), (0, 1), (1, 1), (1, 0)]:
            dr = [(yc - y)/2, (xc - x)/2]
            imgs.append(
                self[fmt[y::2, x::2]].affine(
                    translation=dr, order=order, dims="yx"
                )
            )
        out: ImgArray = np.stack(imgs, axis=newaxis)
        if out.labels is not None:
            del out.labels
            warnings.warn(
                "Output image labels are deleted because it is incompatible with "
                "split_pixel_unit",
                UserWarning,
            )
        out = out[f"{newaxis}={str(angle_order)[1:-1]}"]
        out._set_info(self, new_axes=out.axes)
        out.set_scale(y=self.scale.y*2, x=self.scale.x*2)
        return out

    def stokes(self, *, along: AxisLike = "a") -> dict:
        """
        Generate stocks images from an image stack with polarized images.

        Currently, Degree of Linear Polarization (DoLP) and Angle of
        Polarization (AoP) will be calculated. Those irregular values
        (np.nan, np.inf) will be replaced with 0. Be sure that to calculate
        DoPL correctly background subtraction must be applied beforehand because
        stokes parameter ``s0`` is affected by absolute intensities.

        Parameters
        ----------
        along : AxisLike, default is "a"
            To define which axis is polarization angle axis. Along this axis
            the angle of polarizer must be in order of 0, 45, 90, 135 degree.

        Returns
        -------
        dict
            Dictionaly with keys "dolp" and "aop", which correspond to DoPL and
            AoP respectively.

        Examples
        --------
        Calculate AoP image from the raw image and display them.

            >>> img_pol = img.split_polarization()
            >>> dpol = img_pol.stokes()
            >>> ip.gui.add(img_pol.proj())
            >>> ip.gui.add(dpol.aop.rad2deg())

        References
        ----------
        - https://mavic.ne.jp/indutrialcamera-polarization-inline/
        - Yang, J., Qiu, S., Jin, W., Wang, X., & Xue, F. (2020). Polarization imaging model considering the
          non-ideality of polarizers. Applied optics, 59(2), 306-314.
        - Feng, B., Guo, R., Zhang, F., Zhao, F., & Dong, Y. (2021). Calculation and hue mapping of AoP in
          polarization imaging. May. https://doi.org/10.1117/12.2523643
        """
        new_axes = complement_axes(along, self.axes)
        img0, img45, img90, img135 = (a.as_float().value for a in self.split(along))
        # Stokes parameters
        s0 = (img0 + img45 + img90 + img135)/2
        s1 = img0 - img90
        s2 = img45 - img135

        # Degree of Linear Polarization (DoLP)
        # DoLP is defined as:
        # DoLP = sqrt(s1^2 + s2^2)/s0
        s0[s0==0] = np.inf
        dolp: ImgArray = np.sqrt(s1**2 + s2**2) / s0
        dolp = dolp.view(self.__class__)
        dolp._set_info(self, new_axes=new_axes)
        dolp.set_scale(self)

        # Angle of Polarization (AoP)
        # AoP is usually calculated as psi = 1/2argtan(s1/s2), but this is wrong because
        # left side has range of [0, pi) while right side has range of [-pi/4, pi/4). The
        # correct formulation is:
        #       { 1/2argtan(s2/s1)          (s1>0 and s2>0)
        # AoP = { 1/2argtan(s2/s1) + pi/2   (s1<0)
        #       { 1/2argtan(s2/s1) + pi     (s1>0 and s2<0)
        # But here, np.arctan2 can detect the signs of inputs s1 and s2, so that it returns
        # correct values.
        aop = np.arctan2(s2, s1)/2
        aop: PhaseArray = aop.view(PhaseArray)
        aop._set_info(self, new_axes=new_axes)
        aop.unit = "rad"
        aop.border = (-np.pi/2, np.pi/2)
        aop._fix_border()
        aop.set_scale(self)

        out = dict(dolp=dolp, aop=aop)
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def inpaint(
        self,
        mask: np.ndarray,
        *,
        method: Literal["mean", "biharmonic"] ="biharmonic",
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        Image inpainting.

        Parameters
        ----------
        mask : np.ndarray
            Mask image. The True region will be inpainted.
        method : "mean" or "biharmonic", default is "biharmonic"
            Inpainting method.
        {dims}{update}

        Returns
        -------
        ImgArray
            Inpainted image of same data type.
        """
        if method == "biharmonic":
            from skimage.restoration.inpaint import inpaint_biharmonic
            func = inpaint_biharmonic
        elif method == "mean":
            func = _misc.inpaint_mean
        else:
            raise ValueError(f"Unknown method: {method}")
        return self._apply_dask(
            func,
            c_axes=complement_axes(dims, self.axes),
            args=(mask,),
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def peak_local_max(
        self,
        *,
        min_distance: float = 1.0,
        percentile: float | None = None,
        topn: int = np.inf,
        topn_per_label: int = np.inf,
        exclude_border: bool =True,
        use_labels: bool = True,
        dims: Dims = None
    ) -> MarkerFrame:
        """
        Find local maxima. This algorithm corresponds to ImageJ's 'Find Maxima' but is more flexible.

        Parameters
        ----------
        min_distance : float, default is 1.0
            Minimum distance allowed for each two peaks. This parameter is slightly
            different from that in ``skimage.feature.peak_local_max`` because here float
            input is allowed and every time footprint is calculated.
        percentile : float, optional
            Percentile to compute absolute threshold.
        topn : int, optional
            Maximum number of peaks **for each iteration**.
        topn_per_label : int, default is np.inf
            Maximum number of peaks per label.
        use_labels : bool, default is True
            If use self.labels when it exists.
        {dims}

        Returns
        -------
        MarkerFrame
            DataFrame with columns same as axes of self. For example, if self.axes is "tcyx" then
            return value has "t", "c", "y" and "x" columns, and sub-frame at t=0, c=0 contains all
            the coordinates of peaks in the slice at t=0, c=0.
        """

        # separate spatial dimensions and others
        ndim = len(dims)
        dims_list = list(dims)
        c_axes = complement_axes(dims, self.axes)
        c_axes_list = list(c_axes)

        if isinstance(exclude_border, bool):
            exclude_border = int(min_distance) if exclude_border else False

        thr = None if percentile is None else np.percentile(self.value, percentile)

        import pandas as pd
        from impy.frame import MarkerFrame
        from skimage.feature import peak_local_max

        df_all: list[pd.DataFrame] = []
        for sl, img in self.iter(c_axes, israw=True, exclude=dims):
            # skfeat.peak_local_max overwrite something so we need to give copy of img.
            if use_labels and img.labels is not None:
                labels = xp.asnumpy(img.labels)
            else:
                labels = None

            indices = peak_local_max(
                xp.asnumpy(img),
                footprint=xp.asnumpy(_structures.ball_like(min_distance, ndim)),
                threshold_abs=thr,
                num_peaks=topn,
                num_peaks_per_label=topn_per_label,
                labels=labels,
                exclude_border=exclude_border
            )
            indices = pd.DataFrame(indices, columns=dims_list)
            indices[c_axes_list] = sl
            df_all.append(indices)

        out = pd.concat(df_all, axis=0)
        out = MarkerFrame(out, columns=self.axes, dtype=np.uint16)
        out.set_scale(self)
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def corner_harris(
        self,
        sigma: nDFloat = 1,
        k: float = 0.05,
        *,
        dims: Dims = None
    ) -> ImgArray:
        """
        Calculate Harris response image.

        Parameters
        ----------
        {sigma}
        k : float, optional
            Sensitivity factor to separate corners from edges, typically in range [0, 0.2].
            Small values of k result in detection of sharp corners.
        {dims}

        Returns
        -------
        ImgArray
            Harris response
        """
        from skimage.feature import corner_harris
        return self._apply_dask(
            corner_harris,
            c_axes=complement_axes(dims, self.axes),
            kwargs=dict(k=k, sigma=sigma)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def voronoi(
        self,
        coords: Coords,
        *,
        inf: nDInt | None = None,
        dims: Dims = 2
    ) -> ImgArray:
        """
        Voronoi segmentation of an image.

        Image region labeled with $i$ means that all the points in the region are
        closer to the $i$-th point than any other points.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array-like
            Coordinates of points.
        inf : int, array of int, optional
            Distance to infinity points. If not provided, infinity points are
            placed at 100 times further positions relative to the image shape.
        {dims}

        Returns
        -------
        Label
            Segmentation labels of image.
        """
        from scipy.spatial import Voronoi
        from skimage.measure import grid_points_in_poly
        coords = _check_coordinates(coords, self, dims=self.axes)

        ny, nx = self.sizesof(dims)

        if inf is None:
            infy = ny * 100
            infx = nx * 100
        elif isinstance(inf, int):
            infy = infx = inf
        else:
            infy, infx = inf

        infpoints = np.array(
            [[-infy, -infx],
             [-infy, nx + infx],
             [ny + infy, -infx],
             [ny + infy, nx + infx]],
            dtype=np.float32,
        )

        labels = largest_zeros(self.shape)
        n_label = 1
        for sl, crds in coords.iter(complement_axes(dims, self.axes)):
            input_coords = np.concatenate([crds.values, infpoints], axis=0)
            vor = Voronoi(input_coords)
            for r in vor.regions:
                if all(r0 > 0 for r0 in r):
                    poly = vor.vertices[r]
                    grids = grid_points_in_poly(self.sizesof(dims), poly)
                    labels[sl][grids] = n_label
                    n_label += 1
        self.labels = Label(
            labels, name=self.name+"-label", axes=self.axes, source=self.source
        ).optimize()
        self.labels.set_scale(self)
        return self.labels

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def flood(
        self,
        seeds: Coords,
        *,
        connectivity: int = 1,
        tolerance: float = None,
        dims: Dims = None
    ):
        """
        Flood filling with a list of seed points. By repeating skimage's ``flood`` function,
        this method can perform segmentation of an image.

        Parameters
        ----------
        seeds : MarkerFrame or (N, D) array-like
            Seed points to start flood filling.
        {connectivity}
        tolerance : float, optional
            Intensity deviation within this value will be filled.
        {dims}

        Returns
        -------
        ImgArray
            Labeled image.
        """
        from skimage.morphology import flood
        seeds = _check_coordinates(seeds, self, dims=self.axes)
        labels = largest_zeros(self.shape)
        n_label_next = 1
        for sl, crds in seeds.iter(complement_axes(dims, self.axes)):
            for crd in crds.values:
                crd = tuple(crd)
                if labels[sl][crd] > 0:
                    n_label = labels[sl][crd]
                else:
                    n_label = n_label_next
                    n_label_next += 1
                fill_area = flood(
                    self.value[sl],
                    seed_point=crd,
                    connectivity=connectivity,
                    tolerance=tolerance
                )
                labels[sl][fill_area] = n_label

        self.labels = Label(
            labels, name=self.name+"-label", axes=self.axes, source=self.source
        ).optimize()
        self.labels.set_scale(self)
        return self

    @dims_to_spatial_axes
    def find_sm(
        self,
        sigma: nDFloat = 1.5,
        *,
        method: str = "dog",
        cutoff: float = None,
        percentile: float = 95,
        topn: int = np.inf,
        exclude_border: bool = True,
        dims: Dims = None
    ) -> MarkerFrame:
        """
        Single molecule detection using difference of Gaussian, determinant of Hessian, Laplacian of
        Gaussian or normalized cross correlation method.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of puncta.
        method : str {"dog", "doh", "log", "gaussian", "ncc"}, default is "dog"
            Which filter is used prior to finding local maxima. If "ncc", a Gaussian particle is used as
            the template image.
        cutoff : float, optional
            Cutoff value of filtered image generated by `method`.
        percentile, topn, exclude_border, dims
            Passed to ``peak_local_max()``

        Returns
        -------
        MarkerFrame
            Peaks in uint16 type.

        Examples
        --------
        Track single molecules and view the tracks with napari.

            >>> coords = img.find_sm()
            >>> lnk = coords.link(3, min_dwell=10)
            >>> ip.gui.add(img)
            >>> ip.gui.add(lnk)

        See Also
        --------
        centroid_sm
        refine_sm
        """
        method = method.lower()
        if method in ("dog", "doh", "log"):
            cutoff = 0.0 if cutoff is None else cutoff
            fil_img = getattr(self, method+"_filter")(sigma, dims=dims)
        elif method in ("gaussian",):
            cutoff = -np.inf if cutoff is None else cutoff
            fil_img = getattr(self, method+"_filter")(sigma, dims=dims)
        elif method == "ncc":
            # make template Gaussian
            cutoff = 0.5 if cutoff is None else cutoff
            sigma = np.array(check_nd(sigma, len(dims)))
            shape = tuple((sigma*4).astype(np.int32))
            g = GaussianParticle([(np.array(shape)-1)/2, sigma, 1.0, 0.0])
            template = g.generate(shape)
            # template matching
            fil_img = self.ncc_filter(template)
        else:
            raise ValueError("`method` must be 'dog', 'doh', 'log', 'gaussian' or 'ncc'.")

        fil_img[fil_img<cutoff] = cutoff

        if np.isscalar(sigma):
            min_d = sigma*2
        else:
            min_d = max(sigma)*2
        coords = fil_img.peak_local_max(min_distance=min_d, percentile=percentile,
                                        topn=topn, dims=dims, exclude_border=exclude_border)
        return coords


    @_docs.write_docs
    @dims_to_spatial_axes
    def centroid_sm(
        self,
        coords: Coords = None,
        *,
        radius: nDInt = 4,
        filt: Callable[[ImgArray], bool] = None,
        dims: Dims = None,
        **find_sm_kwargs,
    ) -> MarkerFrame:
        """
        Calculate positions of particles in subpixel precision using centroid.

        Parameters
        ----------
        coords : MarkerFrame or (N, 2) array, optional
            Coordinates of peaks. If None, this will be determined by find_sm.
        radius : int, default is 4.
            Range to calculate centroids. Rectangular image with size 2r+1 x 2r+1 will be send
            to calculate moments.
        filt : callable, optional
            For every slice ``sl``, label is added only when filt(`input`) == True is satisfied.
        find_sm_kwargs : keyword arguments
            Parameters passed to :func:`find_sm`.

        Returns
        -------
        MarkerFrame
            Coordinates of peaks.

        See Also
        --------
        find_sm
        refine_sm
        """
        import pandas as pd
        from ..frame import MarkerFrame
        if coords is None:
            coords = self.find_sm(dims=dims, **find_sm_kwargs)
        else:
            coords = _check_coordinates(coords, self)

        ndim = len(dims)
        filt = check_filter_func(filt)
        radius = np.array(check_nd(radius, ndim))
        shape = self.sizesof(dims)
        out = []
        columns = list(dims)
        c_axes = complement_axes(dims, coords._axes)
        for sl, crds in coords.iter(c_axes):
            centroids = []
            for center in crds.values:
                bbox = _specify_one(center, radius, shape)
                input_img = self.value[sl][bbox]
                if input_img.size == 0 or not filt(input_img):
                    continue

                shift = center - radius
                centroid = _calc_centroid(input_img, ndim) + shift

                centroids.append(centroid.tolist())
            df = pd.DataFrame(centroids, columns=columns)
            df[list(c_axes)] = sl[:-ndim]
            out.append(df)
        if len(out) == 0:
            raise ValueError("No molecule found.")
        out = pd.concat(out, axis=0)

        out = MarkerFrame(out.reindex(columns=list(coords._axes)),
                            columns=str(coords._axes), dtype=np.float32).as_standard_type()
        out.set_scale(coords.scale)

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def edge_grad(
        self,
        sigma: nDFloat = 1.0,
        method: str = "sobel",
        *,
        deg: bool = False,
        dims: Dims = 2
    ) -> PhaseArray:
        """
        Calculate gradient direction using horizontal and vertical edge operation. Gradient direction
        is the direction with maximum gradient, i.e., intensity increase is largest.

        Parameters
        ----------
        {sigma}
        method : str, {"sobel", "farid", "scharr", "prewitt"}, default is "sobel"
            Edge operator name.
        deg : bool, default is True
            If True, degree rather than radian is returned.
        {dims}

        Returns
        -------
        PhaseArray
            Phase image with range [-180, 180) if deg==True, otherwise [-pi, pi).

        Examples
        --------
        1. Profile filament orientation distribution using histogram of edge gradient.
            >>> grad = img.edge_grad(deg=True)
            >>> plt.hist(grad.ravel(), bins=100)
        """
        import skimage.filters

        # Get operator
        methods_ = ["sobel", "farid", "scharr", "prewitt"]
        if method not in methods_:
            raise ValueError(f"`method` must be one of {methods_!r}.")
        op_h = getattr(skimage.filters, method+"_h")
        op_v = getattr(skimage.filters, method+"_v")

        # Start
        c_axes = complement_axes(dims, self.axes)
        if sigma > 0:
            self = self.gaussian_filter(sigma, dims=dims)
        grad_h = self._apply_dask(op_h, c_axes=c_axes)
        grad_v = self._apply_dask(op_v, c_axes=c_axes)
        grad = np.arctan2(-grad_h, grad_v)

        grad = PhaseArray(grad, border=(-np.pi, np.pi))
        grad._fix_border()
        deg and grad.rad2deg()
        return grad

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def hessian_angle(
        self,
        sigma: nDFloat = 1.,
        *,
        deg: bool = False,
        dims: Dims = 2,
    ) -> PhaseArray:
        """
        Calculate filament angles using Hessian's eigenvectors.

        Parameters
        ----------
        {sigma}
        deg : bool, default is False
            If True, returned array will be in degree. Otherwise, radian will be the unit
            of angle.
        {dims}

        Returns
        -------
        ImgArray
            Phase image with range [-90, 90] if ``deg==True``, otherwise [-pi/2, pi/2].

        See Also
        --------
        gabor_angle
        """
        _, eigvec = self.hessian_eig(sigma=sigma, dims=dims)

        fmt = slicer.get_formatter(["dim", "base"])

        arg = -np.arctan2(eigvec[fmt[0, 1]], eigvec[fmt[1, 1]])
        arg = PhaseArray(arg, border=(-np.pi/2, np.pi/2))
        arg._fix_border()
        deg and arg.rad2deg()
        return arg

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def gabor_angle(
        self,
        n_sample: int = 180,
        lmd: float = 5,
        sigma: float = 2.5,
        gamma: float = 1,
        phi: float = 0,
        *,
        deg: bool = False,
        dims: Dims = 2
    ) -> PhaseArray:
        """
        Calculate filament angles using Gabor filter. For all the candidates of angles, Gabor response is
        calculated, and the strongest response is returned as output array.

        Parameters
        ----------
        n_sample : int, default is 180
            Number of `theta`s to calculate. By default, -90, -89, ..., 89 degree are calculated.
        lmd : float, default is 5
            Wave length of Gabor kernel. Make sure that the diameter of the objects you want to detect is
            around `lmd/2`.
        sigma : float, default is 2.5
            Standard deviation of Gaussian factor of Gabor kernel.
        gamma : float, default is 1
            Anisotropy of Gabor kernel, i.e. the standard deviation orthogonal to theta will be sigma/gamma.
        phi : float, default is 0
            Phase offset of harmonic factor of Gabor kernel.
        deg : bool, default is False
            If True, degree rather than radian is returned.
        {dims}

        Returns
        -------
        ImgArray
            Phase image with range [-90, 90) if deg==True, otherwise [-pi/2, pi/2).

        See Also
        --------
        hessian_angle
        """
        from skimage.filters import gabor_kernel
        thetas = np.linspace(0, np.pi, n_sample, False)
        max_ = np.empty(self.shape, dtype=np.float32)
        argmax_ = np.zeros(self.shape, dtype=np.float32) # This is float32 because finally this becomes angle array.

        c_axes = complement_axes(dims, self.axes)
        for i, theta in enumerate(thetas):
            ker = gabor_kernel(1/lmd, theta, 0, sigma, sigma/gamma, 3, phi, dtype=np.complex64)
            out_ = self.as_float()._apply_dask(
                _filters.convolve,
                c_axes=c_axes,
                args=(ker.real,)
            )
            if i > 0:
                where_update = out_ > max_
                max_[where_update] = out_[where_update]
                argmax_[where_update] = i
            else:
                max_ = out_
        argmax_ *= (thetas[1] - thetas[0])
        argmax_[:] = np.pi/2 - argmax_

        argmax_ = PhaseArray(argmax_, border=(-np.pi/2, np.pi/2))
        argmax_._fix_border()
        deg and argmax_.rad2deg()
        return argmax_

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def gabor_filter(
        self,
        lmd: float = 5,
        theta: float = 0,
        sigma: float = 2.5,
        gamma: float = 1,
        phi: float = 0,
        *,
        return_imag: bool = False,
        dims: Dims = 2
    ) -> ImgArray:
        """
        Make a Gabor kernel and convolve it.

        Parameters
        ----------
        lmd : float, default is 5
            Wave length of Gabor kernel. Make sure that the diameter of the objects you want to detect is
            around `lmd/2`.
        theta : float, default is 0
            Orientation of harmonic factor of Gabor kernel in radian (x-directional if `theta==0`).
        {sigma}
        gamma : float, default is 1
            Anisotropy of Gabor kernel, i.e. the standard deviation orthogonal to theta will be sigma/gamma.
        phi : float, default is 0
            Phase offset of harmonic factor of Gabor kernel.
        return_imag : bool, default is False
            If True, a complex image that contains both real and imaginary part of Gabor response is returned.
        {dims}

        Returns
        -------
        ImgArray (dtype is float32 or complex64)
            Filtered image.

        Examples
        --------
        Edge Detection using multi-angle Gabor filtering.
            >>> thetas = np.deg2rad([0, 45, 90, 135])
            >>> out = np.zeros((4,)+img.shape, dtype=np.float32)
            >>> for i, theta in enumerate(thetas):
            >>>     out[i] = img.gabor_filter(theta=theta)
            >>> out = np.max(out, axis=0)
        """
        from skimage.filters import gabor_kernel
        # TODO: 3D Gabor filter
        ker = gabor_kernel(1/lmd, theta, 0, sigma, sigma/gamma, 3, phi).astype(np.complex64)
        if return_imag:
            out = self.as_float()._apply_dask(
                _filters.gabor_filter,
                c_axes=complement_axes(dims, self.axes),
                args=(ker,),
                dtype=np.complex64
            )
        else:
            out = self.as_float()._apply_dask(
                _filters.convolve,
                c_axes=complement_axes(dims, self.axes),
                args=(ker.real,),
                dtype=np.float32
            )
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def fft(
        self,
        *,
        shape: int | Iterable[int] | FftShape = "same",
        shift: bool = True,
        double_precision: bool = False,
        dims: Dims = None
    ) -> ImgArray:
        """
        Fast Fourier transformation. This function returns complex array, which is
        inconpatible with some ImgArray functions.

        Parameters
        ----------
        shape : int, iterable of int, "square" or "same"
            Output shape. Input image is padded or cropped according to this value:
            - integers: padded or cropped to the specified shape.
            - "square": padded to smallest 2^N-long square.
            - "same" (default): no padding or cropping.
        shift : bool, default is True
            If True, call ``np.fft.fftshift`` in the end.
        {double_precision}
        {dims}

        Returns
        -------
        ImgArray
            FFT image.

        See Also
        --------
        local_dft
        """
        axes = [self.axisof(a) for a in dims]
        if shape == "square":
            s = 2**int(np.ceil(np.max(self.sizesof(dims))))
            shape = (s,) * len(dims)
        elif shape == "same":
            shape = None
        else:
            shape = check_nd(shape, len(dims))
        dtype = np.float64 if double_precision else np.float32
        freq = xp.fft.fftn(xp.asarray(self.value, dtype=dtype), s=shape, axes=axes)
        if shift:
            freq[:] = xp.fft.fftshift(freq, axes=axes)
        return xp.asnumpy(freq, dtype=np.complex64)

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def local_dft(
        self,
        key: AxesTargetedSlicer | None = None,
        upsample_factor: nDInt = 1,
        *,
        double_precision: bool = False,
        dims: Dims = None
    ) -> ImgArray:
        r"""
        Local discrete Fourier transformation (DFT). This function will be useful
        for Fourier transformation of small region of an image with a certain
        factor of up-sampling. In general FFT takes :math:`O(N\log{N})` time, much
        faster compared to normal DFT (:math:`O(N^2)`). However, If you are
        interested in certain region of Fourier space, you don't have to calculate
        all the spectra. In this case DFT is faster and less memory comsuming.

            .. warning::
                The result of ``local_dft`` will **NOT** be shifted with
                ``np.fft.fftshift`` because in general the center of arrays are
                unknown. Also, it is easier to understand `x=0` corresponds to the
                center.

        Even whole spectrum is returned, ``local_dft`` may be faster than FFT with
        small and/or non-FFT-friendly shaped image.

        Parameters
        ----------
        key : str
            Key string that specify region to DFT, such as "y=-50:10;x=:80". With
            upsampled spectra, keys corresponds to the coordinate **before**
            up-sampling. If you want certain region, say "x=10:20", this value
            will not change with different ``upsample_factor``.
        upsample_factor : int or array of int, default is 1
            Up-sampling factor. For instance, when ``upsample_factor=10`` a single
            pixel will be expanded to 10 pixels.
        {double_precision}
        {dims}

        Returns
        -------
        ImgArray
            DFT output.

        See Also
        --------
        fft
        """
        ndim = len(dims)
        upsample_factor = check_nd(upsample_factor, ndim)

        # determine how to slice the result of FFT
        if key is None:
            slices = (slice(None),) * ndim
        else:
            slices = solve_slicer(key, Axes(dims), self.shape)
        dtype = np.complex128 if double_precision else np.complex64

        # Calculate exp(-ikx)
        # To minimize floating error, the A term in exp(-2*pi*i*A) should be in the
        # range of 0 <= A < 1.
        exps: list[np.ndarray] = [
            xp.exp(-2j * np.pi * xp.mod(wave_num(sl, s, uf) * xp.arange(s)/s, 1.), dtype=dtype)
            for sl, s, uf in zip(slices, self.sizesof(dims), upsample_factor)
        ]

        # Calculate chunk size for proper output shapes
        out_chunks = np.ones(self.ndim, dtype=np.int64)
        for i, a in enumerate(dims):
            ind = self.axisof(a)
            out_chunks[ind] = exps[i].shape[0]
        out_chunks = tuple(out_chunks)

        return self.as_float()._apply_dask(
            _misc.dft,
            complement_axes(dims, self.axes),
            dtype=np.complex64,
            out_chunks=out_chunks,
            kwargs=dict(exps=exps)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def local_power_spectra(
        self,
        key: AxesTargetedSlicer | None = None,
        upsample_factor: nDInt = 1,
        norm: bool = False,
        *,
        double_precision: bool = False,
        dims: Dims = None
    ) -> ImgArray:
        """
        Return local n-D power spectra of images. See ``local_dft``.

        Parameters
        ----------
        key : str
            Key string that specify region to DFT, such as "y=-50:10;x=:80". With Upsampled spectra, keys
            corresponds to the coordinate **before** up-sampling. If you want certain region, say "x=10:20",
            this value will not change with different ``upsample_factor``.
        upsample_factor : int or array of int, default is 1
            Up-sampling factor. For instance, when ``upsample_factor=10`` a single pixel will be expanded to
            10 pixels.
        norm : bool, default is False
            If True, maximum value of power spectra is adjusted to 1.
        {double_precision}
        {dims}

        Returns
        -------
        ImgArray
            Power spectra

        See Also
        --------
        power_spectra
        """
        freq = self.local_dft(key, upsample_factor=upsample_factor,
                              double_precision=double_precision, dims=dims)
        pw = freq.real**2 + freq.imag**2
        if norm:
            pw /= pw.max()
        return pw

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def ifft(
        self,
        real: bool = True,
        *,
        shift: bool = True,
        double_precision = False,
        dims: Dims = None
    ) -> ImgArray:
        """
        Fast Inverse Fourier transformation. Complementary function with `fft()`.

        Parameters
        ----------
        real : bool, default is True
            If True, only the real part is returned.
        shift : bool, default is True
            If True, call ``np.fft.ifftshift`` at the first.
        {double_precision}
        {dims}

        Returns
        -------
        ImgArray
            IFFT image.
        """
        axes = [self.axisof(a) for a in dims]
        if shift:
            freq = np.fft.ifftshift(self.value, axes=axes)
        else:
            freq = self.value
        dtype = np.complex128 if double_precision else np.complex64
        out = xp.fft.ifftn(
            xp.asarray(freq, dtype=dtype),
            axes=axes
        ).astype(np.complex64)

        if real:
            out = np.real(out)
        return xp.asnumpy(out)

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def power_spectra(
        self,
        shape: int | Iterable[int] | FftShape = "same",
        norm: bool = False,
        zero_norm: bool = False,
        *,
        shift: bool = True,
        double_precision: bool = False,
        dims: Dims = None
    ) -> ImgArray:
        """
        Return n-D power spectra of images, which is defined as:

        .. math::

            P = Re(F[I_{img}])^2 + Im(F[I_{img}])^2

        Parameters
        ----------
        shape : int, iterable of int, "square" or "same"
            Output shape. Input image is padded or cropped according to this value:
            - integers: padded or cropped to the specified shape.
            - "square": padded to smallest 2^N-long square.
            - "same" (default): no padding or cropping.
        norm : bool, default is False
            If True, maximum value of power spectra is adjusted to 1.
        {double_precision}
        shift : bool, default is True
            If True, call ``np.fft.fftshift`` at the first.
        {dims}

        Returns
        -------
        ImgArray
            Power spectra

        See Also
        --------
        local_power_spectra
        """
        freq = self.fft(dims=dims, shape=shape, shift=shift, double_precision=double_precision)
        pw = freq.real**2 + freq.imag**2
        if norm:
            pw /= pw.max()
        pw: ImgArray
        if zero_norm:
            sl = switch_slice(dims, pw.axes, ifin=np.array(pw.shape)//2, ifnot=slice(None))
            pw[sl] = 0
        return pw

    @_docs.write_docs
    @dims_to_spatial_axes
    def radon(
        self,
        degrees: float | Iterable[float],
        *,
        central_axis: AxisLike | Sequence[float] | None = None,
        order: int = 3,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Discrete Radon transformation of 2D or 3D image.

        Radon transformation is a list of projection of a same image from different angles.
        It generates tomographic n-D image slices from (n+1)-D image.

        Parameters
        ----------
        degrees : float or iterable of float
            Rotation angles around the central axis in degrees.
        central_axis : axis-like or sequence of float, optional
            Vector that defines the central axis of rotation.
        {order}{dims}

        Returns
        -------
        ImgArray
            Tomographic image slices. The first spatial axis ("z" for zyx-image) will be dropped.
            If sequence of float is given as ``degrees``, "degree" axis will be newly added at
            the position 0. For instance, if a zyx-image and ``degrees=np.linspace(0, 180, 100)``
            are given, returned image has axes ["degree", "y", "x"].

        See Also
        --------
        iradon
        """
        from dask import array as da, delayed

        params, output_shape, squeeze = _transform.normalize_radon_input(
            self, dims, central_axis, degrees
        )

        # apply spline filter in advance.
        input = self.as_float().spline_filter(order=order)
        delayed_func = delayed(_transform.radon_single)
        tasks = [
            delayed_func(input, p, order=order, output_shape=output_shape)
            for p in params
        ]
        out = xp.stack(da.compute(tasks)[0], axis=0)

        out = xp.asnumpy(out).view(self.__class__)
        out._set_info(self, self.axes.drop(0).insert(0, "degree"))
        out.axes[0].labels = list(degrees)
        if squeeze:
            out = out[0]
        return out

    @_docs.write_docs
    def iradon(
        self,
        degrees: Sequence[float],
        *,
        central_axis: AxisLike = "y",
        degree_axis: AxisLike | None = None,
        height_axis: AxisLike | None = None,
        height: int | None = None,
        window: str = "hamming",
        order: int = 3,
    ) -> ImgArray:
        """
        Inverse Radon transformation (weighted back projection) of a tile series.

        Input array must be a tilt series of 1D or 2D images. They are back-
        projected into a 2D or 3D image with arbitrary height.

        Parameters
        ----------
        degrees : sequence of float
            Projection angles in degree. Length must match the length of the degree
            axis of the input image.
        central_axis : AxisLike, optional
            Axis parallel to the rotation axis.
        degree_axis : AxisLike, optional
            Axis of rotation degree. By default, the first axis will be used.
        height_axis : AxisLike, optional
            Axis that will be used to label the new axis after reconstruction. For
            instance, if input image has axes ``["degree", "y", "x"]`` and
            ``height_axis="z"`` then reconstructed image will have axes
            ``["z", "y", "x"]``. By default, "y" will be used for 2D input or "z" for
            3D input.
        height : int, optional
            Height of reconstruction. By default, size equal to the axis perpendicular
            to the rotation axis will be used.
        window : str, default is "hamming"
            Window function that will be applied to the Fourier domain along the axis
            perpendicular to the rotation axis.
        {order}

        Returns
        -------
        ImgArray
            Reconstruction.

        See Also
        --------
        radon
        """
        from scipy.interpolate import interp1d

        kind = {0: "nearest", 1: "linear", 3: "cubic"}[order]
        central_axis, degree_axis, output_shape, new_axes = _transform.normalize_iradon_input(
            self, central_axis, height_axis, degree_axis, height
        )
        self: ImgArray = np.moveaxis(self, self.axisof(degree_axis), -1)
        filter_func = _transform.get_fourier_filter(self.shape[-2], window)

        interp = partial(interp1d, kind=kind, bounds_error=False, fill_value=0, assume_sorted=True)
        out = self._apply_dask(
            _transform.iradon,
            c_axes=[central_axis],
            kwargs=dict(
                degrees=degrees,
                interp=interp,
                filter_func=filter_func,
                output_shape=output_shape,
            )
        )

        if out.ndim == 3:
            out = np.moveaxis(out, 0, 1)
        out = out[::-1]
        return out._set_info(self, new_axes)

    @check_input_and_output
    def threshold(
        self,
        thr: float | str | ThreasholdMethod = "otsu",
        *,
        along: AxisLike | None = None,
        **kwargs
    ) -> ImgArray:
        """
        Apply thresholding to the image and create a binary image.

        The threshold value can be given with a float or a string that indicates what
        thresholding method will be used.

        Parameters
        ----------
        thr: float or str, optional
            Threshold value, percentage or thresholding algorithm.
        along : AxisLike, optional
            Dimensions that will not share the same threshold. For instance, if
            ``along="c"`` then threshold intensities are determined for every channel.
            If ``thr`` is float, ``along`` will be ignored.
        **kwargs:
            Keyword arguments that will passed to function indicated in 'method'.

        Returns
        -------
        ImgArray
            Boolian array.

        Examples
        --------
        Substitute outliers to 0.
            >>> thr = img.threshold("99%")
            >>> img[thr] = 0
        """
        if self.dtype == bool:
            return self

        import skimage.filters

        if along is None:
            along = "c" if "c" in self.axes else ""

        methods_ = ["isodata", "li", "local", "mean", "min", "minimum", "niblack", "otsu", "sauvola", "triangle", "yen"]

        if isinstance(thr, str) and thr.endswith("%"):
            p = float(thr[:-1].strip())
            out = np.zeros(self.shape, dtype=bool)
            for sl, img in self.iter(along):
                thr = np.percentile(img, p)
                out[sl] = img >= thr

        elif isinstance(thr, str):
            method = thr.lower()
            if method not in methods_:
                raise KeyError(f"{method}\nmethod must be in: {methods_!r}")
            func = getattr(skimage.filters, "threshold_" + method)
            out = np.zeros(self.shape, dtype=bool)
            for sl, img in self.iter(along):
                thr = func(img, **kwargs)
                out[sl] = img >= thr

        elif np.isscalar(thr):
            out = self >= thr
        else:
            raise TypeError(
                "'thr' must be numeric, or str specifying a thresholding method."
            )

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def distance_map(self, *, dims: Dims = None) -> ImgArray:
        """
        Calculate distance map from binary images.
        For instance, ``[1, 1, 1, 0, 0, 0, 1, 1, 1]`` will be converted to
        ``[3, 2, 1, 0, 0, 0, 1, 2, 3]``. Note that returned array will be float in n-D
        images.

        Parameters
        ----------
        {dims}

        Returns
        -------
        ImgArray
            Distance map, the further the brighter
        """
        return self._apply_dask(ndi.distance_transform_edt,
                               c_axes=complement_axes(dims, self.axes)
                               )

    @dims_to_spatial_axes
    @check_input_and_output
    def ncc_filter(
        self,
        template: np.ndarray,
        mode: str = "constant",
        cval: float | str | Callable[[np.ndarray], float] = np.mean,
        *,
        dims: Dims = None
    ) -> ImgArray:
        """
        Template matching using normalized cross correlation (NCC) method. This function is basically
        identical to that in `skimage.feature`, but is optimized for batch processing and improved
        readability.

        Parameters
        ----------
        template : np.ndarray
            Template image. Must be 2 or 3 dimensional.
        {mode}
        cval : float, optional
            Background intensity. If not given, it will calculated as the mean value of
            the original image.
        {dims}

        Returns
        -------
        ImgArray
            Response image with values between -1 and 1.
        """
        template = _check_template(template)
        if callable(cval):
            cval = cval(self)
        cval = _check_bg(self, cval)
        if len(dims) != template.ndim:
            raise ValueError("dims and the number of template dimension don't match.")

        return self.as_float()._apply_dask(
            _filters.ncc_filter,
            c_axes=complement_axes(dims, self.axes),
            args=(template, cval, mode)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def remove_large_objects(self, radius: float = 5, *, dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Remove large objects using opening. Those objects that were not removed by opening
        will be removed in output.

        Parameters
        ----------
        radius : float, optional
            Objects with radius larger than this value will be removed.
        {dims}{update}

        Returns
        -------
        ImgArray
            Image with large objects removed.

        See Also
        --------
        remove_fine_objects
        """
        out = self.copy()
        large_obj = self.opening(radius, dims=dims)
        out.value[large_obj] = 0

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def remove_fine_objects(self, length: float = 10, *, dims: Dims = None, update: bool = False) -> ImgArray:
        """
        Remove fine objects using diameter_opening.

        Parameters
        ----------
        length : float, default is 10
            Objects longer than this will be removed.
        {dims}{update}

        Returns
        -------
        ImgArray
            Image with large objects removed.

        See Also
        --------
        remove_large_objects
        """
        out = self.copy()
        fine_obj = self.diameter_opening(length, connectivity=len(dims))
        large_obj = self.opening(length//2)
        out.value[~large_obj & fine_obj] = 0

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def convex_hull(self, *, dims: Dims = None, update=False) -> ImgArray:
        """
        Compute convex hull image.

        Parameters
        ----------
        {dims}{update}

        Returns
        -------
        ImgArray
            Convex hull image.
        """
        from skimage.morphology import convex_hull_image
        return self._apply_dask(
            convex_hull_image,
            c_axes=complement_axes(dims, self.axes),
            dtype=bool,
        ).astype(bool)

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def skeletonize(self, radius: float = 0, *, dims: Dims = None, update=False) -> ImgArray:
        """
        Skeletonize images. Only works for binary images.

        Parameters
        ----------
        radius : float, optional
            Radius of skeleton. This is achieved simply by dilation of skeletonized results.
        {dims}{update}

        Returns
        -------
        ImgArray
            Skeletonized image.
        """
        if radius >= 1:
            selem = xp.asnumpy(_structures.ball_like(radius, len(dims)))
        else:
            selem = None

        return self._apply_dask(
            _filters.skeletonize,
            c_axes=complement_axes(dims, self.axes),
            args=(selem,),
            dtype=bool
        ).astype(bool)

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def count_neighbors(
        self,
        *,
        connectivity: int | None = None,
        mask: bool = True,
        dims: Dims = None,
    ) -> ImgArray:
        """
        Count the number or neighbors of binary images. This function can be used for cross section
        or branch detection. Only works for binary images.

        Parameters
        ----------
        {connectivity}
        mask : bool, default is True
            If True, only neighbors of pixels that satisfy self==True is returned.
        {dims}

        Returns
        -------
        ImgArray
            uint8 array of the number of neighbors.

        Examples
        --------
            >>> skl = img.threshold().skeletonize()
            >>> edge = skl.count_neighbors()
            >>> np.argwhere(edge == 1) # get coordinates of filament edges.
            >>> np.argwhere(edge >= 3) # get coordinates of filament cross sections.

        """
        ndim = len(dims)
        connectivity = ndim if connectivity is None else connectivity
        selem = ndi.morphology.generate_binary_structure(ndim, connectivity)
        selem[(1,)*ndim] = 0
        out = self.as_uint8()._apply_dask(
            _filters.population,
            c_axes=complement_axes(dims, self.axes),
            args=(selem,)
        )
        if mask:
            out[~self.value] = 0

        return out.astype(np.uint8)

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(only_binary=True)
    def remove_skeleton_structure(
        self,
        structure: Literal["tip", "branch", "cross"] = "tip",
        *,
        connectivity: int = None,
        dims: Dims = None,
        update: bool = False,
    ) -> ImgArray:
        """
        Remove certain structure from skeletonized images.

        Parameters
        ----------
        structure : str, default is "tip"
            What type of structure to remove.
        {connectivity}
        {dims}{update}

        Returns
        -------
        ImgArray
            Processed image.
        """
        out = self.copy()
        neighbor = self.count_neighbors(connectivity=connectivity, dims=dims)
        if structure == "tip":
            sl = neighbor == 1
        elif structure == "branch":
            sl = neighbor > 2
        elif structure == "cross":
            sl = neighbor > 3
        else:
            raise ValueError("`mode` must be one of {'tip', 'branch', 'cross'}.")
        out.value[sl] = 0
        return out

    @dims_to_spatial_axes
    @check_input_and_output(need_labels=True)
    def watershed(
        self,
        coords: MarkerFrame | None = None,
        *,
        connectivity: int = 1,
        input: Literal["self", "distance"] = "distance",
        min_distance: float = 2,
        dims: Dims = None,
    ) -> Label:
        """
        Label segmentation using watershed algorithm.

        Parameters
        ----------
        coords : MarkerFrame, optional
            Returned by such as `peak_local_max()`. Array of coordinates of peaks.
        {connectivity}
        input : str, optional
            What image will be the input of watershed algorithm.

            - "self" ... self is used.
            - "distance" ... distance map of self.labels is used.

        {dims}

        Returns
        -------
        Label
            Updated labels.
        """
        from skimage.segmentation import watershed
        # Prepare the input image.
        if input == "self":
            input_img = self.copy()
        elif input == "distance":
            input_img = self.__class__(self.labels>0, axes=self.axes).distance_map(dims=dims)
        else:
            raise ValueError("'input_' must be either 'self' or 'distance'.")

        if input_img.dtype == bool:
            input_img = input_img.astype(np.uint8)

        input_img.labels = self.labels

        if coords is None:
            coords = input_img.peak_local_max(min_distance=min_distance, dims=dims)

        labels = largest_zeros(input_img.shape)
        shape = self.sizesof(dims)
        n_labels = 0
        c_axes = complement_axes(dims, self.axes)
        markers = np.zeros(shape, dtype=labels.dtype) # placeholder for maxima
        for sl, crd in coords.iter(c_axes):
            # crd.values is (N, 2) array so tuple(crd.values.T.tolist()) is two (N,) list.
            crd = crd.values.T.tolist()
            markers[tuple(crd)] = np.arange(1, len(crd[0])+1, dtype=labels.dtype)
            labels[sl] = watershed(
                -input_img.value[sl],
                markers,
                mask=input_img.labels.value[sl],
                connectivity=connectivity
            )
            labels[sl][labels[sl]>0] += n_labels
            n_labels = labels[sl].max()
            markers[:] = 0 # reset placeholder

        labels = labels.view(Label)
        self.labels = labels.optimize()
        self.labels._set_info(self)
        self.labels.set_scale(self)
        return self.labels

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output(need_labels=True)
    def random_walker(
        self,
        beta: float = 130,
        mode: Literal["cg", "cg_j", "cg_mg", "bf"] = "cg_j",
        tol: float = 1e-3,
        *,
        dims: Dims = None,
    ) -> Label:
        """
        Random walker segmentation. Only wrapped skimage segmentation.

        ``self.labels`` will be segmented and updated inplace.

        Parameters
        ----------
        beta, mode, tol
            see skimage.segmentation.random_walker
        {dims}

        Returns
        -------
        ImgArray
            Relabeled image.
        """
        from skimage.segmentation import random_walker
        c_axes = complement_axes(dims, self.axes)

        for sl, img in self.iter(c_axes, israw=True):
            img.labels[:] = random_walker(
                img.value, img.labels.value, beta=beta, mode=mode, tol=tol
            )

        self.labels._set_info(self)
        return self.labels

    def label_threshold(
        self,
        thr: float | ThreasholdMethod = "otsu",
        filt: Callable[..., bool] | None = None,
        *,
        dims: Dims = None,
        **kwargs,
    ) -> Label:
        """
        Make labels with threshold(). Be sure that keyword argument ``dims`` can be
        different (in most cases for >4D images) between threshold() and label().
        In this function, both function will have the same ``dims`` for simplicity.

        Parameters
        ----------
        thr: float or str or None, optional
            Threshold value, or thresholding algorithm.
        {dims}
        **kwargs:
            Keyword arguments that will passed to function indicated in 'method'.

        Returns
        -------
        Label
            Newly created label.
        """
        labels = self.threshold(thr=thr, **kwargs)
        return self.label(labels, filt=filt, dims=dims)

    def regionprops(
        self,
        properties: Iterable[str] | str = ("mean_intensity",),
        *,
        extra_properties: Iterable[Callable] | None = None,
    ) -> DataDict[str, PropArray]:
        """
        Multi-dimensional region property quantification.

        Run skimage's regionprops() function and return the results as PropArray, so
        that you can access using flexible slicing. For example, if a tcyx-image is
        analyzed with ``properties=("X", "Y")``, then you can get X's time-course profile
        of channel 1 at label 3 by ``prop["X"]["p=5;c=1"]`` or ``prop.X["p=5;c=1"]``.

        Parameters
        ----------
        properties : iterable, optional
            properties to analyze, see ``skimage.measure.regionprops``.
        extra_properties : iterable of callable, optional
            extra properties to analyze, see ``skimage.measure.regionprops``.

        Returns
        -------
        DataDict of PropArray
            Dictionary has keys of properties that are specified by `properties`. Each value
            has the array of properties.

        Examples
        --------
        Measure region properties around single molecules.

            >>> coords = img.centroid_sm()
            >>> img.specify(coords, 3, labeltype="circle")
            >>> props = img.regionprops()
        """
        from skimage.measure import regionprops
        id_axis = "N"
        if isinstance(properties, str):
            properties = (properties,)
        if extra_properties is not None:
            properties = properties + tuple(ex.__name__ for ex in extra_properties)

        if id_axis in self.axes:
            # this dimension will be label
            raise ValueError(f"axis '{id_axis}' is used for label ID in DataFrames.")

        prop_axes = complement_axes(self.labels.axes, self.axes)
        shape = self.sizesof(prop_axes)

        out = DataDict({p: PropArray(
                np.empty((self.labels.max(),) + shape, dtype=np.float32),
                name=self.name+"-prop",
                axes=[id_axis]+prop_axes,
                source=self.source,
                propname=p
                )
                for p in properties
            })

        # calculate property value for each slice
        for sl, img in self.iter(prop_axes, exclude=self.labels.axes):
            props = regionprops(self.labels.value, img, cache=False,
                                      extra_properties=extra_properties)
            label_sl = (slice(None),) + sl
            for prop_name in properties:
                # Both sides have length of id_axis (number of labels) so that values
                # can be correctly substituted.
                out[prop_name][label_sl] = [getattr(prop, prop_name) for prop in props]

        for parr in out.values():
            parr.set_scale(self)
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def lbp(
        self,
        p: int = 12,
        radius: int = 1,
        *,
        method: Literal["default", "ror", "uniform", "var"] = "default",
        dims: Dims = None,
    ) -> ImgArray:
        """
        Local binary pattern feature extraction.

        Parameters
        ----------
        p : int, default is 12
            Number of circular neighbors
        radius : int, default is 1
            Radius of neighbours.
        method : str, optional
            Method to determined the pattern.
        {dims}

        Returns
        -------
        ImgArray
            Local binary pattern image.
        """
        from skimage.feature import local_binary_pattern
        return self._apply_dask(
            local_binary_pattern,
            c_axes=complement_axes(dims),
            args=(p, radius, method)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @check_input_and_output
    def glcm_props(self, distances, angles, radius:int, properties:tuple=None,
                   *, bins:int=None, rescale_max:bool=False, dims: Dims = None) -> ImgArray:
        """
        Compute properties of "Gray Level Coocurrence Matrix". This will take long time
        because of pure Python for-loop.

        Parameters
        ----------
        distances : array_like
            List of pixel pair distance offsets.
        angles : array_like
            List of pixel pair angles in radians.
        radius : int
            Window radius.
        properties : tuple of str
            contrast, dissimilarity, homogeneity, energy, mean, std, asm, max, entropy
        bins : int, optional
            Number of bins.
        rescale_max : bool, default is False
            If True, the contrast of the input image is maximized by multiplying an integer.
        {dims}

        Returns
        -------
        DataDict of ImgArray
            GLCM with additional axes "da", where "d" means distance and "a" means angle.
            If input image has "tzyx" axes then output will have "tzd<yx" axes.

        Examples
        --------
        Plot GLCM's IDM and ASM images
            >>> out = img.glcm_props([1], [0], 3, properties=("idm","asm"))
            >>> out.idm["d=0;<=0"].imshow()
            >>> out.asm["d=0;<=0"].imshow()
        """
        self, bins, rescale_max = _glcm.check_glcm(self, bins, rescale_max)
        if properties is None:
            properties = ("contrast", "dissimilarity", "idm",
                          "asm", "max", "entropy", "correlation")
        c_axes = complement_axes(dims, self.axes)
        distances = np.asarray(distances, dtype=np.uint8)
        angles = np.asarray(angles, dtype=np.float32)
        outshape = self.sizesof(c_axes) + (len(distances), len(angles)) + self.sizesof(dims)
        out = {}
        for prop in properties:
            if isinstance(prop, str):
                out[prop] = np.empty(outshape, dtype=np.float32).view(self.__class__)
            elif callable(prop):
                out[prop.__name__] = np.empty(outshape, dtype=np.float32).view(self.__class__)
            else:
                raise TypeError("properties must be str or callable.")
        out = DataDict(out)
        self = self.pad(radius, mode="reflect", dims=dims)
        for sl, img in self.iter(c_axes):
            propout = _glcm.glcm_props_(img, distances, angles, bins, radius, properties)
            for prop in properties:
                out[prop].value[sl] = propout[prop]

        for k, v in out.items():
            v._set_info(self, new_axes=c_axes+"da"+dims)
        return out


    @same_dtype
    def proj(
        self,
        axis: AxisLike | None = None,
        method: str | Callable = "mean",
        mask = None,
        **kwargs
    ) -> ImgArray:
        """
        Projection along any axis.

        Parameters
        ----------
        axis : str, optional
            Along which axis projection will be calculated. If None, most plausible one will be chosen.
        method : str or callable, default is mean-projection.
            Projection method. If str is given, it will converted to numpy function.
        mask : array-like, optional
            If provided, input image will be converted to np.ma.array and ``method`` will also be interpreted
            as an masked functio if possible.
        **kwargs
            Other keyword arguments that will passed to projection function.

        Returns
        -------
        ImgArray
            Projected image.
        """
        # determine function
        if mask is not None:
            if isinstance(method, str):
                func = getattr(np.ma, method)
            else:
                func = method
        else:
            func = _check_function(method)

        if axis is None:
            axis = find_first_appeared("ztpiac", include=self.axes, exclude="yx")
        elif not hasattr(axis, "__iter__"):
            axis = [axis]
        axisint = tuple(self.axisof(a) for a in axis)
        if func.__module__ == "numpy.ma.core":
            arr = np.ma.array(self.value, mask=mask, dtype=self.dtype)
            if func is np.ma.mean:
                out = func(arr, axis=axisint, dtype=np.float32, **kwargs)
            else:
                out = func(arr, axis=axisint, **kwargs)
        elif func is np.mean:
            out = func(self.value, axis=axisint, dtype=np.float32, **kwargs)
        else:
            out = func(self.value, axis=axisint, **kwargs)

        out = out.view(self.__class__)
        out._set_info(self, self.axes.drop(axisint))
        return out

    @check_input_and_output
    def clip(self, in_range: tuple[int | str, int | str] = ("0%", "100%")) -> ImgArray:
        """
        Saturate low/high intensity using np.clip().

        Parameters
        ----------
        in_range : two scalar values, optional
            range of lower/upper limits, by default (0%, 100%).

        Returns
        -------
        ImgArray
            Clipped image with temporal attribute
        """
        lowerlim, upperlim = _check_clip_range(in_range, self.value)
        out = np.clip(self.value, lowerlim, upperlim)
        out = out.view(self.__class__)
        return out

    @check_input_and_output
    def rescale_intensity(
        self,
        in_range: tuple[int | str, int | str] = ("0%", "100%"),
        dtype = np.uint16
    ) -> ImgArray:
        """
        Rescale the intensity of the image using skimage.exposure.rescale_intensity().

        Parameters
        ----------
        in_range : two scalar values, default is (0%, 100%)
            Range of lower/upper limit.
        dtype : numpy dtype, default is np.uint16
            Output dtype.

        Returns
        -------
        ImgArray
            Rescaled image with temporal attribute
        """
        from skimage.exposure import rescale_intensity

        out = self.view(np.ndarray).astype(np.float32)
        lowerlim, upperlim = _check_clip_range(in_range, self.value)
        out = rescale_intensity(out, in_range=(lowerlim, upperlim), out_range=dtype)
        out = out.view(self.__class__)
        return out

    @check_input_and_output
    def track_drift(
        self,
        along: AxisLike | None = None,
        show_drift: bool = False,
        upsample_factor: int = 10,
        max_shift: nDFloat | None = None,
    ) -> MarkerFrame:
        """Calculate yx-directional drift using the method equivalent to
        ``skimage.registration.phase_cross_correlation``.

        Parameters
        ----------
        along : AxisLike, optional
            Along which axis drift will be calculated.
        show_drift : bool, default is False
            If True, plot the result.
        upsample_factor : int, default is 10
            Up-sampling factor when calculating phase cross correlation.
        max_shift : tuple of float, optional
            Maximum shift in spatial directions.

        Returns
        -------
        MarkerFrame
            DataFrame structure with x,y columns
        """
        from ..frame import MarkerFrame

        if along is None:
            along = find_first_appeared("tpzcia", include=self.axes)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")
        if not isinstance(upsample_factor, int):
            raise TypeError(f"upsample-factor must be integer but got {type(upsample_factor)}")
        if max_shift is not None:
            if np.isscalar(max_shift):
                max_shift = np.full(self.ndim - 1, max_shift)
        result = np.zeros((self.sizeof(along), self.ndim-1), dtype=np.float32)
        c_axes = complement_axes(along, self.axes)
        last_img = None
        img_fft = self.fft(shift=False, dims=c_axes)
        for i, (_, img) in enumerate(img_fft.iter(along)):
            img = xp.asarray(img)
            if last_img is not None:
                result[i] = xp.asnumpy(
                    _corr.subpixel_pcc(
                        last_img,
                        img,
                        max_shifts=max_shift,
                        upsample_factor=upsample_factor,
                    )[0]
                )
                last_img = img
            else:
                last_img = img

        result = MarkerFrame(np.cumsum(result, axis=0), columns=c_axes)
        if show_drift:
            from ._utils import _plot as _plt
            _plt.plot_drift(result)

        result.index.name = str(along)
        return result

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def drift_correction(
        self,
        shift: Coords | None = None,
        ref: ImgArray | Any | None = None,
        *,
        zero_ave: bool = True,
        along: AxisLike | None = None,
        max_shift: nDFloat | None = None,
        order: int = 1,
        mode: str = "constant",
        cval: float = 0,
        dims: Dims = 2,
        update: bool = False,
    ) -> ImgArray:
        """Drift correction using iterative Affine translation.

        If translation vectors  ``shift`` is not given, then it will be determined using
        ``track_drift`` method of ImgArray.

        Parameters
        ----------
        shift : DataFrame or (N, D) array, optional
            Translation vectors. If DataFrame, it must have columns named with all the symbols
            contained in ``dims``.
        ref : ImgArray or slicer, optional
            The reference n-D image to determine drift, if ``shift`` was not given. This
            parameter can be a slicer, which will be used to slice the image to make a
            reference.
        zero_ave : bool, default is True
            If True, average shift will be zero.
        along : AxisLike, optional
            Along which axis drift will be corrected.
        {dims}{update}
        affine_kwargs :
            Keyword arguments that will be passed to ``warp``.

        Returns
        -------
        ImgArray
            Corrected image.

        Examples
        --------
        Drift correction of multichannel image using the first channel as the reference.
        >>> img.drift_correction(ref=img["c=0"])
        """

        if along is None:
            along = find_first_appeared("tpzcia", include=self.axes, exclude=dims)
        elif len(along) != 1:
            raise ValueError("`along` must be single character.")

        if shift is None:
            # determine 'ref'
            if ref is None:
                ref = self
            elif not isinstance(ref, ImgArray):
                ref = self[ref]
            if ref.axes != [along] + dims:
                _c_axes = complement_axes([along] + dims, str(ref.axes))
                fmt = slicer.get_formatter(_c_axes)
                out = np.empty_like(self)
                for idx in product(*(range(ref.sizeof(a)) for a in _c_axes)):
                    sl = fmt[idx]
                    out[sl] = self[sl].drift_correction(
                        ref=ref[sl], zero_ave=zero_ave, along=along, dims=dims,
                        update=update, order=order, mode=mode, cval=cval,
                    )
                return out

            shift = ref.track_drift(along=along, max_shift=max_shift).values

        else:
            shift = np.asarray(shift, dtype=np.float32)
            if (self.sizeof(along), self.ndim) != shift.shape:
                raise ValueError("Wrong shape of `shift`.")

        if zero_ave:
            shift = shift - np.mean(shift, axis=0)

        out = xp.empty(self.shape)
        t_index = self.axisof(along)
        ndim = len(dims)
        mx = np.eye(ndim + 1, dtype=np.float32) # Affine transformation matrix
        input_img = self.spline_filter(mode=mode, order=order)
        for sl, img in input_img.iter(complement_axes(dims, self.axes)):
            mx[:-1, -1] = -shift[sl[t_index]]
            out[sl] = _transform.warp(img, mx, mode=mode, cval=cval, order=order, prefilter=False)

        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    def estimate_sigma(self, *, squeeze: bool = True, dims: Dims = None) -> PropArray | float:
        """
        Wavelet-based estimation of Gaussian noise.

        Parameters
        ----------
        squeeze : bool, default is True
            If True and output can be converted to a scalar, then convert it.
        {dims}

        Returns
        -------
        PropArray or float
            Estimated standard deviation. sigma["t=0;c=1"] means the estimated value of
            image slice at t=0 and c=1.
        """
        from skimage.restoration import estimate_sigma
        c_axes = complement_axes(dims, self.axes)
        out = self._apply_dask(
            estimate_sigma,
            c_axes=c_axes,
            drop_axis=dims
        )

        if out.ndim == 0 and squeeze:
            out = out.item()
        else:
            out = PropArray(
                out, dtype=np.float32, name=self.name, axes=c_axes, propname="estimate_sigma"
            )
            out._set_info(self, new_axes=c_axes)
        return out

    @_docs.write_docs
    @dims_to_spatial_axes
    def center_of_mass(self, dims: Dims = None) -> PropArray:
        """
        Calculate the center of mass of the image.

        Parameters
        ----------
        {dims}

        Returns
        -------
        PropArray
            Center of mass. Axes will be the input axes minus ``dims``, plus a new axis
            ``dim`` at the first position, which represents the dimensions of the
            results.
        """
        c_axes = complement_axes(dims, self.axes)
        out_shape = (len(dims), ) + self.sizesof(c_axes)
        axis = Axis("dim")
        axis.labels = dims
        out_axes = [axis] + c_axes
        out = np.empty(out_shape, dtype=np.float32)
        for sl, img in self.iter(c_axes, israw=True, exclude=dims):
            out[(slice(None),) + sl] = ndi.center_of_mass(img)
        out = PropArray(
            out, dtype=np.float32, name=self.name, axes=out_axes, propname="center_of_mass"
        )
        out._set_info(self, new_axes=out_axes)
        return out

    @dims_to_spatial_axes
    @check_input_and_output
    def pad(
        self,
        pad_width: int | tuple[int, int] | Sequence[tuple[int, int]],
        *,
        mode: PaddingMode = "constant",
        dims: Dims = None,
        **kwargs
    ) -> ImgArray:
        """
        Pad image only for spatial dimensions.

        Parameters
        ----------
        pad_width, mode, **kwargs :
            See documentation of np.pad().
        dims : int or str, optional
            Which dimension to pad.

        Returns
        -------
        ImgArray
            Padded image.

        Examples
        --------
        Suppose ``img`` has zyx-axes.

        1. Padding 5 pixels in zyx-direction:
            >>> img.pad(5)
        2. Padding 5 pixels in yx-direction:
            >>> img.pad(5, dims="yx")
        3. Padding 5 pixels in yx-direction and 2 pixels in z-direction:
            >>> img.pad([(5,5), (4,4), (4,4)])
        4. Padding 10 pixels in z-(-)-direction and 5 pixels in z-(+)-direction.
            >>> img.pad([(10, 5)], dims="z")
        """
        pad_width = _misc.make_pad(pad_width, dims, self.axes, **kwargs)
        padimg = np.pad(self.value, pad_width, mode, **kwargs).view(self.__class__)
        return padimg

    @same_dtype(asfloat=True)
    @check_input_and_output
    def pad_defocus(self, kernel, *, depth: int = 3, width: int = 6, bg: float = None) -> ImgArray:
        """
        Make a z-directional padded image by defocusing the original image. This padding is
        useful when applying FFT to a 3D image.

        Parameters
        ----------
        kernel : 0-, 1- or 3-dimensional array.
            If 0- (scalar) or 1-dimensional array was given, this is interpreted as standard
            deviation of Gaussian kernel. If 3-dimensional array was given, this is directly
            used as convolution kernel. Other dimension will raise ValueError.
        depth : int, default is 3
            Depth of defocusing. For an image with z-axis size L, then output image will have
            size L + 2*depth.
        width : int, default is 6
            Width of defocusing. For an image with yx-shape (M, N), then output image will have
            shape (M * 2*width, N + 2*width).
        bg : float, optional
            Background intensity. If not given, it will calculated as the minimum value of
            the original image.

        Returns
        -------
        ImgArray
            Padded image.

        Examples
        --------
        depth = 2

        .. code-block::

            ----|   |----| o |--     o ... center of kernel
            ----| o |----|   |--
            ++++|   |++++|___|++  <- the upper edge of original image
            ++++|___|+++++++++++

        """
        bg = _check_bg(self, bg)

        if np.isscalar(kernel):
            kernel = np.array([kernel]*3)
        else:
            kernel = np.asarray(kernel)

        if kernel.ndim <= 1:
            def filter_func(img):
                return xp.asnumpy(_filters.gaussian_filter(img, kernel, mode="constant", cval=bg))
            dz, dy, dx = kernel*3 # 3-sigma

        elif kernel.ndim == 3:
            kernel = kernel.astype(np.float32)
            kernel = kernel / np.sum(kernel)
            def filter_func(img):
                return xp.asnumpy(_filters.convolve(img, kernel, mode="constant", cval=bg))
            dz, dy, dx = np.array(kernel.shape)//2
        else:
            raise ValueError("`kernel` only take 0, 1, 3 dimensional array as an input.")

        pad_width = [(depth, depth), (width, width), (width, width)]
        padimg = self.pad(pad_width, mode="constant", constant_values=bg, dims="zyx")
        out = np.copy(padimg.value)
        # convolve psf
        z_edge0 = slice(None, depth, None)
        z_mid = slice(depth, -depth, None)
        z_edge1 = slice(-depth, None, None)
        y_edge0 = x_edge0 = slice(None, width, None)
        y_mid = slice(width, -width, None)
        y_edge1 = x_edge1 = slice(-width, None, None)

        for sl, img in padimg.iter(complement_axes("zyx", self.axes)):
            out_ = out[sl]
            out_[z_edge0] = filter_func(img[:depth+dz ])[z_edge0]
            out_[z_edge1] = filter_func(img[-depth-dz:])[z_edge1]
            out_[z_mid, y_edge0] = filter_func(img[:, :width+dy ])[z_mid, y_edge0]
            out_[z_mid, y_edge1] = filter_func(img[:, -width-dy:])[z_mid, y_edge1]
            out_[z_mid, y_mid, x_edge0] = filter_func(img[:, :, :width+dx ])[z_mid, y_mid, x_edge0]
            out_[z_mid, y_mid, x_edge1] = filter_func(img[:, :, -width-dx:])[z_mid, y_mid, x_edge1]

        return out.view(self.__class__)


    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def wiener(
        self,
        psf: np.ndarray | Callable[[tuple[int, ...]], np.ndarray],
        lmd: float = 0.1,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        r"""
        Classical wiener deconvolution. This algorithm has the serious ringing problem
        if parameters are set to wrong values.

        :math:`F[Y_{res}] = \frac{F[Y_{obs}] \cdot \bar{H}}{|H|^2 + \lambda}`

        :math:`Y_{obs}`: observed image;
        :math:`Y_{res}`: restored image;
        :math:`H` : FFT of point spread function (PSF);

        Parameters
        ----------
        psf : ndarray or callable
            Point spread function. If a function is given, `psf(shape)` will be
            called to generate the PSF.
        lmd : float, default is 0.1
            Constant value used in the deconvolution. See Formulation below.
        {dims}{update}

        Returns
        -------
        ImgArray
            Deconvolved image.


        See Also
        --------
        lucy
        lucy_tv
        """
        if lmd <= 0:
            raise ValueError(f"lmd must be positive, but got: {lmd}")

        psf_ft, psf_ft_conj = _deconv.check_psf(
            self.sizesof(dims), tuple(self.scale[axis] for axis in dims), psf,
        )

        return self._apply_dask(
            _deconv.wiener,
            c_axes=complement_axes(dims, self.axes),
            args=(psf_ft, psf_ft_conj, lmd)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def lucy(
        self,
        psf: np.ndarray | Callable[[tuple[int, ...]], np.ndarray],
        niter: int = 50,
        eps: float = 1e-5,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        """
        Deconvolution of N-dimensional image, using Richardson-Lucy's algorithm.

        Parameters
        ----------
        psf : ndarray or callable
            Point spread function. If a function is given, `psf(shape)` will be
            called to generate the PSF.
        niter : int, default is 50.
            Number of iterations.
        eps : float, default is 1e-5
            During deconvolution, division by small values in the convolve image
            of estimation and PSF may cause divergence. Therefore, division by
            values under `eps` is substituted to zero.
        {dims}{update}

        Returns
        -------
        ImgArray
            Deconvolved image.

        See Also
        --------
        lucy_tv
        wiener
        """

        psf_ft, psf_ft_conj = _deconv.check_psf(
            self.sizesof(dims), tuple(self.scale[axis] for axis in dims), psf,
        )

        return self._apply_dask(
            _deconv.richardson_lucy,
            c_axes=complement_axes(dims, self.axes),
            args=(psf_ft, psf_ft_conj, niter, eps)
        )

    @_docs.write_docs
    @dims_to_spatial_axes
    @same_dtype(asfloat=True)
    @check_input_and_output
    def lucy_tv(
        self,
        psf: np.ndarray | Callable[[tuple[int, ...]], np.ndarray],
        max_iter: int = 50,
        lmd: float = 1e-3,
        tol: float = 1e-3,
        eps: float = 1e-5,
        *,
        dims: Dims = None,
        update: bool = False
    ) -> ImgArray:
        r"""
        Deconvolution of N-dimensional image, using Richardson-Lucy's algorithm with
        total variance regularization (so called RL-TV algorithm). The TV regularization
        factor at pixel position :math:`x`, :math:`F_{reg}(x)`, is calculated as:

        .. math::

            F_{reg}(x) = \frac{1}{1-\lambda \cdot div(\frac{grad(I(x)}{|grad(I(x))|})}

        (:math:`I(x)`: image, :math:`\lambda`: constant)

        and this factor is multiplied for every estimation made in each iteration.

        Parameters
        ----------
        psf : ndarray or callable
            Point spread function. If a function is given, `psf(shape)` will be
            called to generate the PSF.
        max_iter : int, default is 50.
            Maximum number of iterations.
        lmd : float, default is 1e-3
            The constant lambda of TV regularization factor.
        tol : float, default is 1e-3
            Iteration stops if regularized absolute summation is lower than this
            value.

            :math:`\frac{\sum_{x}|I'(x) - I(x)|}{\sum_{x}|I(x)|}`

            (:math:`I'(x)`: estimation of :math:`k+1`-th iteration, :math:`I(x)`:
            estimation of :math:`k`-th iteration)

        eps : float, default is 1e-5
            During deconvolution, division by small values in the convolve image of
            estimation and PSF may cause divergence. Therefore, division by values
            under ``eps`` is substituted to zero.
        {dims}{update}

        Returns
        -------
        ImgArray
            Deconvolved image.

        References
        ----------
        - Dey, N., Blanc-Féraud, L., Zimmer, C., Roux, P., Kam, Z., Olivo-Marin, J. C.,
          & Zerubia, J. (2004). 3D microscopy deconvolution using Richardson-Lucy algorithm
          with total variation regularization (Doctoral dissertation, INRIA).

        See Also
        --------
        lucy
        wiener
        """
        if lmd <= 0:
            raise ValueError(
                "In Richadson-Lucy with total-variance-regularization, "
                "parameter `lmd` must be positive."
            )
        psf_ft, psf_ft_conj = _deconv.check_psf(
            self.sizesof(dims), tuple(self.scale[axis] for axis in dims), psf
        )

        return self._apply_dask(
            _deconv.richardson_lucy_tv,
            c_axes=complement_axes(dims, self.axes),
            args=(psf_ft, psf_ft_conj, max_iter, lmd, tol, eps)
        )


    @overload
    def mean(
        self, axis: Literal[None] = None, dtype: DTypeLike | None = None,
        out: Any = None, keepdims: Literal[False] = False, *, where: Any = _NoValue,
    ) -> np.number:
        ...

    @overload
    def mean(
        self, axis: int | AxisLike | Iterable[int] | Iterable[AxisLike],
        dtype: DTypeLike | None = None, out: Any = None, keepdims: bool = False,
        *, where: Any = _NoValue,
    ) -> ImgArray:
        ...

    def mean(
        self,
        axis=None,
        dtype: DTypeLike | None = None,
        out: Any | None = None,
        keepdims: bool = False,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Mean value of the array along a given axis."""
        if self.dtype.kind in "ui" and dtype is None:
            dtype = np.float32  # to avoid using np.float64
        return super().mean(axis=axis, dtype=dtype, out=out, keepdims=keepdims, where=where)

    @overload
    def std(
        self, axis: Literal[None] = None, dtype: DTypeLike | None = None,
        out: Any | None = None, keepdims: Literal[False] = False, *, where: Any = _NoValue,
    ) -> np.number:
        ...

    @overload
    def std(
        self, axis: int | AxisLike | Iterable[int] | Iterable[AxisLike],
        dtype: DTypeLike | None = None, out: Any | None = None, keepdims: bool = False,
        *, where: Any = _NoValue,
    ) -> ImgArray:
        ...

    def std(
        self,
        axis=None,
        dtype: DTypeLike = None,
        out: None = None,
        ddof: int = 0,
        keepdims: bool = False,
        *,
        where: np.ndarray = _NoValue,
    ):
        """Standard deviation of the array along a given axis."""
        if self.dtype.kind in "ui" and dtype is None:
            dtype = np.float32  # to avoid using np.float64
        return super().std(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims, where=where)


def _check_coordinates(coords, img: ImgArray, dims: Dims = None):
    from impy.frame import MarkerFrame

    if not isinstance(coords, MarkerFrame):
        coords = np.asarray(coords)
        if coords.ndim == 1:
            coords = coords.reshape(1, -1)
        elif coords.ndim != 2:
            raise ValueError("Input cannot be interpreted as coordinate(s).")
        if dims is None:
            ndim = coords.shape[1]
            if ndim == img.ndim:
                dims = img.axes
            else:
                dims = complement_axes("c", img.axes)[-ndim:]
        coords = MarkerFrame(coords, columns=dims, dtype=np.uint16)
        coords.set_scale(img)

    return coords

def _check_function(func):
    if isinstance(func, str):
        func = getattr(np, func, None)
    if callable(func):
        return func
    else:
        raise TypeError("Must be one of numpy methods or callable object.")

def _check_bg(img: ImgArray, bg):
    # determine bg
    if bg is None:
        bg = img.min()
    elif isinstance(bg, str) and bg.endswith("%"):
        bg = np.percentile(img.value, float(bg[:-1]))
    elif not np.isscalar(bg):
        raise TypeError("Wrong type of `bg`.")
    return bg

def _check_template(template):
    if not isinstance(template, np.ndarray):
        raise TypeError(f"`template` must be np.ndarray, but got {type(template)}")
    elif template.ndim not in (2, 3):
        raise ValueError("`template must be 2 or 3 dimensional.`")
    template = np.asarray(template).astype(np.float32, copy=True)
    return template

def _calc_centroid(img: np.ndarray, ndim: int) -> np.ndarray:
    from skimage.measure import moments
    mom = moments(img, order=1)
    centroid = np.array([mom[(0,)*i + (1,) + (0,)*(ndim-i-1)]
                        for i in range(ndim)]) / mom[(0,)*ndim]
    return centroid

def _check_clip_range(in_range, img) -> tuple[float, float]:
    """
    Called in clip_outliers() and rescale_intensity().
    """
    lower, upper = in_range
    if isinstance(lower, str) and lower.endswith("%"):
        lower = float(lower[:-1])
        lowerlim = np.percentile(img, lower)
    elif lower is None:
        lowerlim = np.min(img)
    else:
        lowerlim = float(lower)

    if isinstance(upper, str) and upper.endswith("%"):
        upper = float(upper[:-1])
        upperlim = np.percentile(img, upper)
    elif upper is None:
        upperlim = np.max(img)
    else:
        upperlim = float(upper)

    if lowerlim >= upperlim:
        raise ValueError(f"lowerlim is larger than upperlim: {lowerlim} >= {upperlim}")

    return lowerlim, upperlim

def _specify_one(center, radius, shape:tuple) -> tuple[slice]:
    sl = tuple(slice(max(0, xc-r), min(xc+r+1, sh), None)
                        for xc, r, sh in zip(center, radius, shape))
    return sl

def check_filter_func(f):
    if f is None:
        return lambda x: True
    elif not callable(f):
        raise TypeError("`filt` must be callable.")
    return f


def wave_num(sl: slice, s: int, uf: int) -> np.ndarray:
    """
    A function that makes wave number vertical vector. Returned vector will
    be [k/s, (k + 1/uf)/s, (k + 2/uf)/s, ...] (where k = sl.start)

    Parameters
    ----------
    sl : slice
        Slice that specify which part of the image will be transformed.
    s : int
        Size along certain dimension.
    uf : int
        Up-sampling factor of certain dimension.
    """
    start = 0 if sl.start is None else sl.start
    stop = s if sl.stop is None else sl.stop

    if sl.start and sl.stop and start < 0 and stop > 0:
        # like "x=-5:5"
        pass
    else:
        if -s < start < 0:
            start += s
        elif not 0 <= start < s:
            raise ValueError(f"Invalid value encountered in slice {sl}.")
        if -s < stop <= 0:
            stop += s
        elif not 0 < stop <= s:
            raise ValueError(f"Invalid value encountered in slice {sl}.")

    n = stop - start
    return xp.linspace(start, stop, n*uf, endpoint=False)[:, np.newaxis]
