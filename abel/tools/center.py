# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from .math import fit_gaussian
import warnings
from scipy.ndimage import center_of_mass, shift
from scipy.optimize import minimize
# testing strings with Python 2 and 3 compatibility
from six import string_types

from abel import _deprecated, _deprecate


def find_origin(IM, method='image_center', axes=(0, 1), verbose=False,
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

    axes : int or tuple of int
        find origin coordinates: ``0`` (vertical), or ``1`` (horizontal), or
        ``(0, 1)`` (both vertical and horizontal).

    Returns
    -------
    out : (float, float)
        coordinates of the origin of the image in the (row, column) format.
        For coordinates not in **axes**, the center of the image is returned.

    """
    return func_method[method](IM, axes, verbose=verbose, **kwargs)


def center_image(IM, method='com', odd_size=True, square=False, axes=(0, 1),
                 crop='maintain_size', order=3, verbose=False,
                 center=_deprecated, **kwargs):
    """
    Center image with the custom value or by several methods provided in
    :func:`find_origin()` function.

    Parameters
    ----------
    IM : 2D np.array
        The image data.

    method : str or tuple of float
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

        See :func:`set_center` for examples.

    axes : int or tuple of int
        center image with respect to axis ``0`` (vertical), ``1`` (horizontal),
        or both axes ``(0, 1)`` (default). When specifying an explicit origin
        in **method**, unused coordinates can also be passed as ``None``, for
        example, ``method=(row, None)`` or ``method=(None, col)``.

    order : int
        interpolation order, see :func:`set_center` for details.

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
            trim = diff // 2
            if trim > 0:
                IM = IM[trim: -trim]  # remove even number of rows off each end
            if diff % 2:
                IM = IM[: -1]  # remove one additional row

        else:
            # make rows == cols, check row oddness
            if odd_size and rows % 2 == 0:
                IM = IM[:-1, :]
                rows -= 1
            xs = (cols - rows) // 2
            IM = IM[:, xs:-xs]

        rows, cols = IM.shape

    # origin is in (row, column) format!
    if isinstance(method, string_types):
        origin = find_origin(IM, method=method, axes=axes, verbose=verbose,
                             **kwargs)
    else:
        origin = method

    centered_data = set_center(IM, origin=origin, crop=crop, axes=axes,
                               order=order, verbose=verbose)
    return centered_data


def set_center(data, origin, crop='maintain_size', axes=(0, 1), order=3,
               verbose=False, center=_deprecated):
    """
    Move image origin to mid-point of image (``rows // 2, cols // 2``).

    Parameters
    ----------
    data : 2D np.array
        the image data

    origin : tuple of float
        (row, column) coordinates of the image origin.
        Coordinates set to ``None`` are ignored.

    crop : str
        determines how the image should be cropped. The options are:

        ``maintain_size`` (default)
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

        Examples:

        .. plot:: tools/crop_options.py

    axes : int or tuple of int
        center image with respect to axis ``0`` (vertical), ``1`` (horizontal),
        or both axes ``(0, 1)`` (default).

    order : int
        interpolation order (0–5, default is 3) for centering with fractional
        **origin**. Lower orders work faster; **order** = 0 (also implied for
        integer **origin**) means a whole-pixel shift without interpolation and
        is much faster.

    verbose : bool
        print some information for debugging

    Returns
    -------
    out : 2D np.array
        centered image
    """
    if center is not _deprecated:
        _deprecate('abel.tools.center.set_center() '
                   'argument "center" is deprecated, use "origin" instead.')
        origin = center

    def center_of(a):
        """ Indices of array center """
        return np.array(a.shape) // 2

    shape = data.shape
    if verbose:
        print('Original shape', shape, 'and origin', tuple(origin))
    center = center_of(data)

    # remove axes with "None" coordinates, preprocess origin
    if isinstance(axes, int):
        axes = [axes]
    axes = set(axes)
    origin = np.array(origin, dtype=object)  # (for None, int or float)
    subpixel = np.zeros(2)
    origin_ = [None, None]
    for a in [0, 1]:
        if origin[a] is None:
            axes.discard(a)
        else:
            # to absolute coordinates
            if origin[a] < 0:
                origin[a] += shape[a]
            if order:
                # split integer and fractional parts
                i = int(origin[a])
                origin[a], subpixel[a] = i, origin[a] - i
            else:
                # round to whole pixels
                origin[a] = int(round(origin[a]))
            # complement (from the other edge)
            origin_[a] = shape[a] - 1 - origin[a]
    # don't interpolate for whole-pixels shifts
    if np.all(subpixel == 0):
        order = 0
    if verbose:
        print('Centering axes', tuple(axes), 'using order', order)

    # Note: current scipy.ndimage.shift() implementation erases fractional edge
    # pixels, so we wrap it with padding and cropping. Once this behavior is
    # corrected, our code can be cleaned up.

    if crop == 'maintain_size':
        delta = [0, 0]
        if order:  # fractional shift
            for a in axes:
                if origin[a] is not None:
                    delta[a] = center[a] - (origin[a] + subpixel[a])
            out = shift(np.pad(data, 1, 'constant'),
                        delta, order=order)[1:-1, 1:-1]  # (see note above)
        else:  # whole-pixel shift
            src = [slice(None), slice(None)]  # for the source region
            dst = [slice(None), slice(None)]  # for the destination region
            for a in axes:
                delta[a] = center[a] - origin[a]
                # gaps for positive and negative shifts wrt edges
                dpos = max(0, delta[a])
                dneg = max(0, -delta[a])
                # corresponding regions
                src[a] = slice(dneg, shape[a] - dpos)
                dst[a] = slice(dpos, shape[a] - dneg)
            out = np.zeros_like(data)
            out[tuple(dst)] = data[tuple(src)]
        if verbose:
            print('Shifted by', tuple(delta))
            print('Output shape', out.shape,
                  'centered at', tuple(center_of(out)))
        return out
    # for other crop options, first do subpixel shift
    # size will change to add/remove the shifted fractional pixel parts
    if order:
        if verbose:
            print('Subpixel shift by', tuple(-subpixel))
        # (see the note above about padding on both sides and cropping)
        data = shift(np.pad(data, 1, 'constant'),
                     -subpixel, order=order)[:-1, :-1]
        # shift origin or cut unused pixels
        cut = [slice(None), slice(None)]
        for a in [0, 1]:
            if subpixel[a]:
                if crop == 'valid_region':
                    cut[a] = slice(1, -1)  # cut both fractional ends
                    origin_[a] -= 1
                else:
                    origin[a] += 1
                # (shape is not used below, thus not updated here)
            else:
                cut[a] = slice(1, None)  # cut empty (not shifted) pixel
        data = data[tuple(cut)]
    if crop == 'valid_region':
        src = [slice(None), slice(None)]
        for a in axes:
            # distance to the closest edge
            d = min(origin[a], origin_[a])
            # crop symmetrically around the origin
            src[a] = slice(origin[a] - d, origin[a] + d + 1)
        out = data[tuple(src)].copy()  # (independent data, as in other cases)
        if verbose:
            print('Output cropped to', out.shape,
                  'centered at', tuple(center_of(out)))
        return out
    elif crop == 'maintain_data':
        pad = [(0, 0), (0, 0)]
        for a in axes:
            # distance to the farthest edge
            d = max(origin[a], origin_[a])
            # pad to symmetrize around the origin
            pad[a] = (d - origin[a], d - origin_[a])
        out = np.pad(data, pad, 'constant')
        if verbose:
            print('Output padded to', out.shape,
                  'centered at', tuple(center_of(out)))
        return out
    else:
        raise ValueError('Invalid crop option "{}".'.format(crop))


def find_origin_by_center_of_mass(IM, axes=(0, 1), verbose=False,
                                  round_output=False, **kwargs):
    """
    Find image origin by calculating its center of mass.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    round_output : bool
        if ``True``, the coordinates are rounded to integers;
        otherwise they are floats.

    axes : int or tuple
        find origin coordinates: ``0`` (vertical), or ``1`` (horizontal), or
        ``(0, 1)`` (both vertical and horizontal).

    Returns
    -------
    origin : (float, float)
        (row, column)
    """
    origin = list(center_of_mass(IM))

    # reset unneeded coordinates
    if isinstance(axes, int):
        axes = [axes]
    for a in set([0, 1]) - set(axes):
        origin[a] = IM.shape[a] // 2
    origin = tuple(origin)

    if verbose:
        to_print = "Center of mass at {}".format(origin)

    if round_output:
        origin = (round(origin[0]), round(origin[1]))
        if verbose:
            to_print += " ... round to {}".format(origin)

    if verbose:
        print(to_print)

    return origin


def find_origin_by_convolution(IM, axes=(0, 1), projections=False, **kwargs):
    """
    Find the image origin as the maximum of autoconvolution of its projections
    along each axis.

    Code from the ``linbasex`` juptyer notebook.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    projections : bool
        if this parameter is ``True``, the autoconvoluted projections along
        both axes will be returned after the origin.

    axes : int or tuple
        find origin coordinates: ``0`` (vertical), or ``1`` (horizontal), or
        ``(0, 1)`` (both vertical and horizontal).

    Returns
    -------
    origin : (float, float)
        (row, column)

        `or` (row, column), conv_0, conv_1
    """
    if isinstance(axes, int):
        axes = [axes]

    conv = [None, None]
    origin = [IM.shape[0] // 2, IM.shape[1] // 2]
    for a in axes:
        # projection along the other axis
        proj = IM.sum(axis=1 - a)
        # autoconvolute projections
        conv[a] = np.convolve(proj, proj, mode='full')
        # take the first max, should there be several equal maxima
        origin[a] = np.argmax(conv[a]) / 2
    origin = tuple(origin)

    if projections:
        return origin, conv[0], conv[1]
    else:
        return origin


def find_origin_by_center_of_image(data, axes=(0, 1), verbose=False, **kwargs):
    """
    Find image origin simply as its center, from its dimensions.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    axes : int or tuple
        has no effect

    Returns
    -------
    origin : (int, int)
        (row, column)

    """
    return (data.shape[0] // 2, data.shape[1] // 2)


def find_origin_by_gaussian_fit(IM, axes=(0, 1), verbose=False,
                                round_output=False, **kwargs):
    """
    Find image origin by fitting the summation along rows and columns of the
    data to two 1D Gaussian functions.

    Parameters
    ----------
    IM : numpy 2D array
        image data

    axes : int or tuple
        find origin coordinates: ``0`` (vertical), or ``1`` (horizontal), or
        ``(0, 1)`` (both vertical and horizontal).

    round_output : bool
        if ``True``, the coordinates are rounded to integers;
        otherwise they are floats.

    Returns
    -------
    origin : (float, float)
        (row, column)
    """
    if isinstance(axes, int):
        axes = [axes]

    origin = [IM.shape[0] // 2, IM.shape[1] // 2]
    for a in axes:
        # sum along the other axis
        proj = np.sum(IM, axis=1 - a)
        # find gaussian center
        origin[a] = fit_gaussian(proj)[1]
    origin = tuple(origin)

    if verbose:
        to_print = "Gaussian origin at {}".format(origin)

    if round_output:
        origin = (round(origin[0]), round(origin[1]))
        if verbose:
            to_print += " ... round to {}".format(origin)

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

    r2 = rows // 2
    c2 = cols // 2
    sw2 = min(slice_width // 2, r2, c2)

    rmin, rmax = radial_range

    # vertical slice
    top = IM[:r2, c2-sw2:c2+sw2+1].sum(axis=1)
    bottom = IM[r2 + rows % 2:, c2-sw2:c2+sw2+1].sum(axis=1)

    # horizontal slice
    left = IM[r2-sw2:r2+sw2+1, :c2].sum(axis=0)
    right = IM[r2-sw2:r2+sw2+1, c2 + cols % 2:].sum(axis=0)

    return (top[::-1][rmin:rmax], bottom[rmin:rmax],
            left[::-1][rmin:rmax], right[rmin:rmax])


def find_origin_by_slice(IM, axes=(0, 1), slice_width=10, radial_range=(0, -1),
                         axis=_deprecated, **kwargs):
    """
    Find the image origin by comparing opposite sides.

    Parameters
    ----------
    IM : 2D np.array
        the image data

    slice_width : integer
        Sum together this number of rows (cols) to improve signal, default 10.

    radial_range: tuple
        (rmin, rmax): radial range ``[rmin:rmax]`` for slice profile
        comparison.

    axes : int or tuple
        find origin coordinates: ``0`` (vertical), or ``1`` (horizontal), or
        ``(0, 1)`` (both vertical and horizontal).

    Returns
    -------
    origin : (float, float)
        (row, column)

    """
    if axis is not _deprecated:
        _deprecate('abel.tools.center.find_origin_by_slice() '
                   'argument "axis" is deprecated, use "axes" instead.')
        axes = axis

    def _align(offset, sliceA, sliceB):
        """
        Intensity difference between an axial slice and its shifted opposite.
        """
        # always shift to the left (towards center)
        if offset < 0:
            diff = shift(sliceA, offset) - sliceB
        else:
            diff = sliceA - shift(sliceB, -offset)
        fvec = (diff**2).sum()
        return fvec

    if isinstance(axes, int):
        axes = [axes]

    rows, cols = IM.shape

    r2 = (rows - 1) / 2
    c2 = (cols - 1) / 2
    top, bottom, left, right = axis_slices(IM, radial_range, slice_width)

    xyoffset = [0.0, 0.0]
    # determine shift to align both slices
    # limit shift to +- 20 pixels
    initial_shift = [0.1, ]

    # vertical axis
    if 0 in axes:
        fit = minimize(_align, initial_shift, args=(top, bottom),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[0] = -float(fit['x']) / 2  # x1/2 for image shift
        else:
            raise RuntimeError("fit failure: axis 0, zero shift set", fit)

    # horizontal axis
    if 1 in axes:
        fit = minimize(_align, initial_shift, args=(left, right),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[1] = -float(fit['x']) / 2  # x1/2 for image shift
        else:
            raise RuntimeError("fit failure: axis 1, zero shift set", fit)

    # this is the (row, col) shift to align the slice profiles
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
    if square:
        _deprecate('Argument "square" has no effect and is deprecated.')
    return find_origin(IM, center, verbose, **kwargs)


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
