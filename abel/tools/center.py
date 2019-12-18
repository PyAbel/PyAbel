# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .math import fit_gaussian
import warnings
from scipy.ndimage import center_of_mass
from scipy.ndimage.interpolation import shift
from scipy.optimize import minimize
# testing strings with Python 2 and 3 compatibility
from six import string_types

from abel import _deprecated, _deprecate


def find_origin(IM, method='image_center', square=False, verbose=False,
                **kwargs):
    """
    Find the coordinates of image origin, using the specified method.

    Parameters
    ----------
    IM : 2D np.array
        image data

    method : str
        determines how the origin should be found. The options are:

        ``image_center``
            the center of the image is used as the origin. The trivial result.
        ``com``
            the origin is found as the center of mass.
        ``convolution``
            the origin is found as the maximum of autoconvolution of the image
            projections along each axis.
        ``gaussian``
            the origin is extracted by a fit to a Gaussian function.
            This is probably only appropriate if the data resembles a
            gaussian.
        ``slice``
            the image is broken into slices, and these slices compared
            for symmetry.

    square : bool
        if ``True``, returned image will have a square shape

    Returns
    -------
    out : (float, float)
        coordinates of the origin of the image in the (row, column) format

    """
    return func_method[method](IM, verbose=verbose, **kwargs)


def center_image(IM, method='com', odd_size=True, square=False, axes=(0, 1),
                 crop='maintain_size', verbose=False, center=_deprecated,
                 **kwargs):
    """
    Center image with the custom value or by several methods provided in
    :func:`find_origin()` function.

    Parameters
    ----------
    IM : 2D np.array
        The image data.

    method : tuple or str
        either a tuple (float, float), the coordinate of the origin of the
        image in the (row, column) format, or a string to specify an automatic
        centering method:

        ``image_center``
            the center of the image is used as the origin. The trivial result.
        ``com``
            the origin is found as the center of mass.
        ``convolution``
            the origin is found as the maximum of autoconvolution of the image
            projections along each axis.
        ``gaussian``
            the origin is extracted from a fit to a Gaussian function.
            This is probably only appropriate if the data resembles a
            gaussian.
        ``slice``
            the image is broken into slices, and these slices compared for
            symmetry.

    odd_size : boolean
        if ``True``, the returned image will contain an odd number of columns.
        Most of the transform methods require this, so it's best to set this
        to ``True`` if the image will subsequently be Abel-transformed.

    square : bool
        if ``True``, the returned image will have a square shape.

    crop : str
        determines how the image should be cropped. The options are:

        ``maintain_size``
            return image of the same size. Some regions of the original image
            may be lost, and some regions may be filled with zeros.
        ``valid_region``
            return the largest image that can be created without padding.
            All of the returned image will correspond to the original image.
            However, portions of the original image will be lost.
            If you can tolerate clipping the edges of the image, this is
            probably the method to choose.
        ``maintain_data``
            the image will be padded with zeros such that none of the original
            image will be cropped.

    axes : int or tuple
        center image with respect to axis ``0`` (vertical), ``1`` (horizontal),
        or both axes ``(0, 1)`` (default).

    Returns
    -------
    out : 2D np.array
        centered image
    """
    if center is not _deprecated:
        _deprecate('abel.tools.center.center_image() '
                   'argument "center" is deprecated, use "method" instead.')
        method = center

    rows, cols = IM.shape

    if odd_size and cols % 2 == 0:
        # drop rightside column
        IM = IM[:, :-1]
        rows, cols = IM.shape

    if square and rows != cols:
        # make rows == cols, but maintain approx. center
        if rows > cols:
            diff = rows - cols
            trim = diff//2
            if trim > 0:
                IM = IM[trim: -trim]  # remove even number of rows off each end
            if diff % 2:
                IM = IM[: -1]  # remove one additional row

        else:
            # make rows == cols, check row oddness
            if odd_size and rows % 2 == 0:
                IM = IM[:-1, :]
                rows -= 1
            xs = (cols - rows)//2
            IM = IM[:, xs:-xs]

        rows, cols = IM.shape

    # origin is in (row, column) format!
    if isinstance(method, string_types):
        origin = find_origin(IM, method=method, verbose=verbose, **kwargs)
    else:
        origin = method

    centered_data = set_center(IM, origin=origin, crop=crop, axes=axes,
                               verbose=verbose)
    return centered_data


def set_center(data, origin, crop='maintain_size', axes=(0, 1), verbose=False,
               center=_deprecated):
    """
    Move image origin to mid-point of image.

    Parameters
    ----------
    data : 2D np.array
        the image data

    origin : tuple
        (row, column) coordinates of the image origin

    crop : str
        determines how the image should be cropped. The options are:

        ``maintain_size``
            return image of the same size. Some regions of the original image
            may be lost and some regions may be filled with zeros.
        ``valid_region``
            return the largest image that can be created without padding.
            All of the returned image will correspond to the original image.
            However, portions of the original image will be lost.
            If you can tolerate clipping the edges of the image, this is
            probably the method to choose.
        ``maintain_data``
            the image will be padded with zeros such that none of the original
            image will be cropped.

    axes : int or tuple
        center image with respect to axis ``0`` (vertical), ``1`` (horizontal),
        or both axes ``(0, 1)`` (default).

    verbose : bool
        ``True``: print diagnostics
    """
    if center is not _deprecated:
        _deprecate('abel.tools.center.set_center() '
                   'argument "center" is deprecated, use "origin" instead.')
        origin = center

    old_shape = data.shape
    old_center = data.shape[0] // 2, data.shape[1] // 2

    origin = list(origin)
    if origin[0] < 0:
        origin[0] += data.shape[0]
    if origin[1] < 0:
        origin[1] += data.shape[1]

    delta0 = old_center[0] - origin[0]
    delta1 = old_center[1] - origin[1]

    if crop == 'maintain_data':
        # pad the image so that the origin can be moved without losing any of
        # the original data

        # we need to pad the image with zeros before using the shift() function
        shift0, shift1 = (None, None)
        if delta0 != 0:
            shift0 = 1 + int(np.abs(delta0))
        if delta1 != 0:
            shift1 = 1 + int(np.abs(delta1))

        container = np.zeros((data.shape[0] + shift0, data.shape[1] + shift1),
                             dtype=data.dtype)

        area = container[:, :]
        if shift0:
            if delta0 > 0:
                area = area[:-shift0, :]
            else:
                area = area[shift0:, :]
        if shift1:
            if delta1 > 0:
                area = area[:, :-shift1]
            else:
                area = area[:, shift1:]
        area[:, :] = data[:, :]
        data = container
        delta0 += np.sign(delta0) * shift0 / 2.0
        delta1 += np.sign(delta1) * shift1 / 2.0
    if verbose:
        print("delta = ({0}, {1})".format(delta0, delta1))

    if isinstance(axes, int):
        if axes == 0:
            centered_data = shift(data, (delta0, 0))
        elif axes == 1:
            centered_data = shift(data, (0, delta1))
        else:
            raise ValueError("axes value not 0, or 1")
    else:
        centered_data = shift(data, (delta0, delta1))

    if crop == 'maintain_data':
        # pad the image so that the origin can be moved
        # without losing any of the original data
        return centered_data
    elif crop == 'maintain_size':
        return centered_data
    elif crop == 'valid_region':
        # crop to region containing data
        shift0, shift1 = (None, None)
        if delta0 != 0:
            shift0 = 1 + int(np.abs(delta0))
        if delta1 != 0:
            shift1 = 1 + int(np.abs(delta1))
        return centered_data[shift0:-shift0, shift1:-shift1]
    else:
        raise ValueError("Invalid crop method!!")


def find_origin_by_center_of_mass(IM, verbose=False, round_output=False,
                                  **kwargs):
    """
    Find image origin by calculating its center of mass.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    round_output : bool
        if ``True``, the coordinates are rounded to integers;
        otherwise they are floats.

    Returns
    -------
    origin : tuple
        (row, column)
    """
    origin = center_of_mass(IM)

    if verbose:
        to_print = "Center of mass at ({0}, {1})".format(origin[0], origin[1])

    if round_output:
        origin = (round(origin[0]), round(origin[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(origin[0], origin[1])

    if verbose:
        print(to_print)

    return origin


def find_origin_by_convolution(IM, **kwargs):
    """
    Find the image origin as the maximum of autoconvolution of its projections
    along each axis.

    Code from the ``linbasex`` juptyer notebook.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    projections : any
        if this parameter is present, the autoconvoluted projections along
        both axes will be returned after the origin.

    Returns
    -------
    origin : tuple
        (row, column)

        `or` (row, column), conv_0, conv_1
    """
    # projection along axis=0 of image (rows)
    QL_raw0 = IM.sum(axis=1)
    # projection along axis=1 of image (cols)
    QL_raw1 = IM.sum(axis=0)

    # autoconvolute projections
    conv_0 = np.convolve(QL_raw0, QL_raw0, mode='full')
    conv_1 = np.convolve(QL_raw1, QL_raw1, mode='full')

    # Take the first max, should there be several equal maxima.
    # 10May16 - axes swapped - check this
    origin = (np.argmax(conv_0) / 2, np.argmax(conv_1) / 2)

    if "projections" in kwargs.keys():
        return origin, conv_0, conv_1
    else:
        return origin


def find_origin_by_center_of_image(data, verbose=False, **kwargs):
    """
    Find image origin simply as its center, from its dimensions.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    Returns
    -------
    origin : tuple
        (row, column)

    """
    return (data.shape[0] // 2, data.shape[1] // 2)


def find_origin_by_gaussian_fit(IM, verbose=False, round_output=False,
                                **kwargs):
    """
    Find image origin by fitting the summation along rows and columns of the
    data to two 1D Gaussian functions.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    round_output : bool
        if ``True``, the coordinates are rounded to integers;
        otherwise they are floats.

    Returns
    -------
    origin : tuple
        (row, column)
    """
    x = np.sum(IM, axis=0)
    y = np.sum(IM, axis=1)
    xc = fit_gaussian(x)[1]
    yc = fit_gaussian(y)[1]
    origin = (yc, xc)

    if verbose:
        to_print = "Gaussian origin at ({0}, {1})".format(origin[0], origin[1])

    if round_output:
        origin = (round(origin[0]), round(origin[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(origin[0], origin[1])

    if verbose:
        print(to_print)

    return origin


def axis_slices(IM, radial_range=(0, -1), slice_width=10):
    """
    Returns vertical and horizontal slice profiles, summed across slice_width.

    Parameters
    ----------
    IM : 2D np.array
        image data

    radial_range: tuple of float
        (rmin, rmax) range to limit data

    slice_width : int
        width of the image slice, default 10 pixels

    Returns
    -------
    top, bottom, left, right : 1D np.arrays shape (rmin:rmax, 1)
        image slices oriented in the same direction
    """
    rows, cols = IM.shape   # image size

    r2 = rows // 2 + rows % 2
    c2 = cols // 2 + cols % 2
    sw2 = slice_width // 2

    rmin, rmax = radial_range

    # vertical slice
    top = IM[:r2, c2-sw2:c2+sw2].sum(axis=1)
    bottom = IM[r2 - rows % 2:, c2-sw2:c2+sw2].sum(axis=1)

    # horizontal slice
    left = IM[r2-sw2:r2+sw2, :c2].sum(axis=0)
    right = IM[r2-sw2:r2+sw2, c2 - cols % 2:].sum(axis=0)

    return (top[::-1][rmin:rmax], bottom[rmin:rmax],
            left[::-1][rmin:rmax], right[rmin:rmax])


def find_origin_by_slice(IM, slice_width=10, radial_range=(0, -1),
                         axis=(0, 1), **kwargs):
    """
    Find the image origin by comparing opposite sides: vertical (``axis=0``)
    and/or horizontal slice (``axis=1``) profiles. To find both coordinates,
    use ``axis=(0, 1)``.

    Parameters
    ----------
    IM : 2D np.array
        the image data

    slice_width : integer
        Sum together this number of rows (cols) to improve signal, default 10.

    radial_range: tuple
        (rmin, rmax): radial range ``[rmin:rmax]`` for slice profile
        comparison.

    axis : int or tuple
        Find origin coordinates: ``axis=0`` (vertical), or ``axis=1``
        (horizontal), or ``axis=(0, 1)`` (both vertical and horizontal).

    Returns
    -------
    origin : tuple
        (row, column)

    """

    def _align(offset, sliceA, sliceB):
        """intensity difference between an axial slice and its shifted opposite.
        """
        diff = shift(sliceA, offset) - sliceB
        fvec = (diff**2).sum()
        return fvec

    if not isinstance(axis, (list, tuple)):
        # if the user supplies an int, make it into a 1-element list:
        axis = [axis]

    rows, cols = IM.shape

    r2 = rows // 2
    c2 = cols // 2
    top, bottom, left, right = axis_slices(IM, radial_range, slice_width)

    xyoffset = [0.0, 0.0]
    # determine shift to align both slices
    # limit shift to +- 20 pixels
    initial_shift = [0.1, ]

    # vertical axis
    if 0 in axis:
        fit = minimize(_align, initial_shift, args=(top, bottom),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[0] = -float(fit['x']) / 2  # x1/2 for image shift
        else:
            if verbose:
                print("fit failure: axis = 0, zero shift set")
                print(fit)

    # horizontal axis
    if 1 in axis:
        fit = minimize(_align, initial_shift, args=(left, right),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[1] = -float(fit['x']) / 2  # x1/2 for image shift
        else:
            raise RuntimeError("fit failure: axis = 1, zero shift set", fit)

    # this is the (row, col) shift to align the slice profiles
    xyoffset = tuple(xyoffset)

    return r2 - xyoffset[0], c2 - xyoffset[1]


func_method = {
    "image_center": find_origin_by_center_of_image,
    "com": find_origin_by_center_of_mass,
    "convolution": find_origin_by_convolution,
    "gaussian": find_origin_by_gaussian_fit,
    "slice": find_origin_by_slice
}


# Deprecated functions

def find_center(IM, center='image_center', square=False, verbose=False,
                **kwargs):
    """Deprecated function. Use :func:`find_origin` instead."""
    _deprecate('abel.tools.center.find_center() '
               'is deprecated, use abel.tools.center.find_origin() instead.')
    return find_origin(IM, center, square, verbose, **kwargs)


def find_center_by_center_of_mass(IM, verbose=False, round_output=False,
                                  **kwargs):
    """Deprecated function. Use :func:`find_origin_by_center_of_mass`
    instead."""
    _deprecate('abel.tools.center.find_center_by_center_of_mass() '
               'is deprecated, use '
               'abel.tools.center.find_origin_by_center_of_mass() instead.')
    return find_origin_by_center_of_mass(IM, verbose, round_output, **kwargs)


def find_center_by_convolution(IM, **kwargs):
    """Deprecated function. Use :func:`find_origin_by_convolution` instead."""
    _deprecate('abel.tools.center.find_center_by_convolution() '
               'is deprecated, use '
               'abel.tools.center.find_origin_by_convolution() instead.')
    return find_origin_by_convolution(IM, **kwargs)


def find_center_by_center_of_image(data, verbose=False, **kwargs):
    """Deprecated function. Use :func:`find_origin_by_center_of_image`
    instead."""
    _deprecate('abel.tools.center.find_center_by_center_of_image() '
               'is deprecated, use '
               'abel.tools.center.find_origin_by_center_of_image() instead.')
    return find_origin_by_center_of_image(data, verbose, **kwargs)


def find_center_by_gaussian_fit(IM, verbose=False, round_output=False,
                                **kwargs):
    """Deprecated function. Use :func:`find_origin_by_gaussian_fit` instead."""
    _deprecate('abel.tools.center.find_center_by_gaussian_fit() '
               'is deprecated, use '
               'abel.tools.center.find_origin_by_gaussian_fit() instead.')
    return find_origin_by_gaussian_fit(IM, verbose, round_output, **kwargs)


def find_image_center_by_slice(IM, slice_width=10, radial_range=(0, -1),
                               axis=(0, 1), **kwargs):
    """Deprecated function. Use :func:`find_origin_by_slice` instead."""
    _deprecate('abel.tools.center.find_image_center_by_slice() '
               'is deprecated, use '
               'abel.tools.center.find_origin_by_slice() instead.')
    return find_origin_by_slice(IM, slice_width, radial_range, axis, **kwargs)
