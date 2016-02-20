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


def center_image(IM, center='com', verbose=False):
    """ Center image with the custom value or by several methods provided in `find_center` function

    Parameters
    ----------
    IM : 2D np.array
       The image data.

    center : (float, float) or
       - (float, float): coordinate of the center of the image in the (y,x) format (row, column)
       - str: use provided method in `find_center` function to determine the center of the image

    Returns
    -------
    out : 2D np.array
       Centered image
    """

    # center is in y,x (row column) format!
    if isinstance(center, str) or isinstance(center, unicode):
        center = find_center(IM, center, verbose=verbose)

    centered_data = set_center(IM, center, verbose=verbose)
    return centered_data


def set_center(data, center, crop='maintain_size', verbose=True):
    c0, c1 = center
    if isinstance(c0, (int, long)) and isinstance(c1, (int, long)):
        warnings.warn('Integer center detected, but not respected.'
                      'treating center as float and interpolating!')
        # need to include code here to treat integer centers
        # probably can use abel.tools.symmetry.center_image_asym(),
        # but this function lacks the ability to set the vertical center

    old_shape = data.shape
    old_center = data.shape[0]/2.0, data.shape[1]/2.0

    delta0 = old_center[0] - center[0]
    delta1 = old_center[1] - center[1]

    if crop == 'maintain_data':
        # pad the image so that the center can be moved without losing any of the original data
        # we need to pad the image with zeros before using the shift() function
        shift0, shift1 = (None, None)
        if delta0 != 0:
            shift0 = 1 + int(np.abs(delta0))
        if delta1 != 0:
            shift1 = 1 + int(np.abs(delta1))

        container = np.zeros((data.shape[0]+shift0, data.shape[1]+shift1),
                              dtype = data.dtype)

        area = container[:,:]
        if shift0:
            if delta0 > 0:
                area = area[:-shift0,:]
            else:
                area = area[shift0:,:]
        if shift1:
           if delta1 > 0:
                area = area[:,:-shift1]
           else:
                area = area[:,shift1:]
        area[:,:] = data[:,:]
        data = container
        delta0 += np.sign(delta0)*shift0/2.0
        delta1 += np.sign(delta1)*shift1/2.0
    if verbose:
        print("delta = ({0}, {1})".format(delta0, delta1))

    centered_data = shift(data, (delta0, delta1))

    if crop == 'maintain_data':
        # pad the image so that the center can be moved
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



def find_center(IM, method='image_center', verbose=True, **kwargs):
    """
    Paramters
    ---------
    IM : 2D np.array
      image data

    method: str
      valid method:
        - image_center
        - com
        - gaussian

    Returns
    -------
    out: (float, float)
      coordinate of the center of the image in the (y,x) format (row, column)

    """
    return func_method[method](IM, verbose=verbose, **kwargs)


def find_center_by_center_of_mass(IM, verbose=True, round_output=False,
                                  **kwargs):
    """
    Find image center by calculating its center of mass
    """
    com = center_of_mass(IM)
    center = com[0], com[1]

    if verbose:
        to_print = "Center of mass at ({0}, {1})".format(center[0], center[1])

    if round_output:
        center = (round(center[0]), round(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print)

    return center


def find_center_by_center_of_image(data, verbose=True, **kwargs):
    return (data.shape[1] // 2 + data.shape[1] % 2,
            data.shape[0] // 2 + data.shape[0] % 2)




def find_center_by_gaussian_fit(IM, verbose=True, round_output=False,
                                **kwargs):
    """
    Find image center by fitting the summation along x and y axis of the data to two 1D Gaussian function
    """
    x = np.sum(IM, axis=0)
    y = np.sum(IM, axis=1)
    xc = fit_gaussian(x)[1]
    yc = fit_gaussian(y)[1]
    center = (yc, xc)

    if verbose:
        to_print = "Gaussian center at ({0}, {1})".format(center[0], center[1])

    if round_output:
        center = (round(center[0]), round(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print)

    return center

func_method = {
    "image_center": find_center_by_center_of_image,
    "com": find_center_by_center_of_mass,
    "gaussian": find_center_by_gaussian_fit,
    "slice": find_image_center_by_slice
}


def axis_slices(IM, radial_range=(0, -1), slice_width=10):
    """returns vertical and horizontal slice profiles, summed across slice_width.

    Paramters
    ---------
    IM : 2D np.array
      image data

    radial_range: tuple floats
      (rmin, rmax) range to limit data

    slice_width : integer
      width of the image slice, default 10 pixels

    Returns
    -------
    top, bottom, left, right : 1D np.arrays shape (rmin:rmax, 1)
      image slices oriented in the same direction

    """
    rows, cols = IM.shape   # image size

    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2
    sw2 = slice_width/2

    rmin, rmax = radial_range

    # vertical slice
    top = IM[:r2, c2-sw2:c2+sw2].sum(axis=1)
    bottom = IM[r2 - rows % 2:, c2-sw2:c2+sw2].sum(axis=1)

    # horizontal slice
    left = IM[r2-sw2:r2+sw2, :c2].sum(axis=0)
    right = IM[r2-sw2:r2+sw2, c2 - cols % 2:].sum(axis=0)

    return (top[::-1][rmin:rmax], bottom[rmin:rmax],
            left[::-1][rmin:rmax], right[rmin:rmax])


def find_image_center_by_slice(IM, slice_width=10, radial_range=(0, -1),
                               axis=(0, 1)):
    """ Center image by comparing opposite side, vertical (axis=0) and/or
        horizontal slice (axis=1) profiles, both axis=(0,1)..

    Parameters
    ----------
    IM : 2D np.array
       The image data.

    slice_width : integer
       Sum together this number of rows (cols) to improve signal, default 10.

    radial_range: tuple
       (rmin,rmax): radial range [rmin:rmax] for slice profile comparison.

    axis : integer or tuple
       Center with along axis = 0 (vertical), or 1 (horizontal), or both (0,1).

    Returns
    -------
    IMcenter : 2D np.array
       Centered image

    (vertical_shift, horizontal_shift) : tuple of floats
       (axis=0 shift, axis=1 shift)

    """

    def _align(offset, sliceA, sliceB):
        """intensity difference between an axial slice and its shifted opposite.
        """
        diff = shift(sliceA, offset) - sliceB
        fvec = (diff**2).sum()
        return fvec

    rows, cols = IM.shape

    if cols % 2 == 0:
        # drop rightside column, and bottom row to make odd size
        IM = IM[:-1, :-1]
        rows, cols = IM.shape

    top, bottom, left, right = axis_slices(IM, radial_range, slice_width)

    xyoffset = [0.0, 0.0]
    # determine shift to align both slices
    # limit shift to +- 20 pixels
    initial_shift = [0.1, ]

    # y-axis
    if (type(axis) is int and axis == 0) or \
            (type(axis) is tuple and axis[0] == 0):
        fit = minimize(_align, initial_shift, args=(top, bottom),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[0] = -float(fit['x'])/2  # x1/2 for image center shift
        else:
            print("fit failure: axis = 0, zero shift set")
            print(fit)

    # x-axis
    if (type(axis) is int and axis == 1) or \
            (type(axis) is tuple and axis[1] == 1):
        fit = minimize(_align, initial_shift, args=(left, right),
                       bounds=((-50, 50), ), tol=0.1)
        if fit["success"]:
            xyoffset[1] = -float(fit['x'])/2   # x1/2 for image center shift
        else:
            print("fit failure: axis = 1, zero shift set")
            print(fit)

    # this is the (y, x) shift to align the slice profiles
    xyoffset = tuple(xyoffset)

    IM_centered = shift(IM, xyoffset)  # center image

    return IM_centered, xyoffset
