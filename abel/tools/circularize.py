# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.optimize import leastsq

import abel

# see https://github.com/PyAbel/PyAbel/issues/186 for discussion


def circularize_image(IM, method="argmax", center=None, radial_range=None,
                      zoom=1, smooth=1.0e-7, nslices=32, ref_angle=None, 
                      inverse=False, return_correction=False):
    """
    Remove radial distortion from a velocity-map-image through radial scaling
    of the Newton-rings to enforce circularity.

    The image is divided into angular slices, adjacent radial intensity
    profile slices compared to give a radial scaling factor that best
    aligns common structure.

    The resultant radial scaling factor vs angle is interpolated to apply the
    correction to the whole image grid. The image is remapped to the
    correct cartesian grid.

    Parameters
    ----------
    IM : numpy 2D array

    method: str
        method to align slice profiles,
        "argmax" - compare intensity-profile.argmax() of each radial slice.
        "lsq" - determine radial scaling factor by minimizing the
                intensity-prfoile with an adjacent slice intensity-profile.

    center: str, or None
        pre-center image using PyAbel centering methods,
        "com", "convolution", "gaussian", "image_center", "slice"

    radial_range: tuple, or None
        limit slice comparison to the radial range (rmin, rmax).

    zoom: float
        nmimage.zoom image before analysis. This may help improve the low
        radius slice profile.

    smooth: float
        smoothing for spline interpolation of the determined radial scaling
        factor vs angle

    nslices: int
        divide the image into nslices. Ideally, this should be a divisor of
        the image column width.

    ref_angle: None or float
        reference angle for which radial coordinate is unchanged.
        None uses numpy.mean(radial scale factors)

    inverse: bool
        inverse Abel transform the *polar* image, to remove the background
        intensity. This helps follow a given peak intensityr, for all angles,
        when the anisotropy parameter is large.

    return_correction: bool
        additional outputs, see below

    Returns
    -------
    IMcirc : numpy 2D array, same size as input
        Circularized imput image

    (if return_correction is True)
    slice_angles: numpy 1D array
        Mid-point angle (radians) of image slice

    radial_corrections: numpy 1D array
        radial correction scale factors, at each angle

    spline_function: numpy func()
        spline function used to evaluate the radial correction at each angle

    """

    if zoom > 1:
        # zoom may improve the low radii angular resolution
        IM = ndimage.zoom(IM, zoom)

    if center is not None:
        # convenience function for the case image is not centered
        IM = abel.tools.center.center_image(IM, center=center)

    # map image into polar coordinates - much easier to slice
    # cartesian (Y, X) -> polar (Radius, Theta)
    polarIM, radial_coord, angle_coord =\
                           abel.tools.polar.reproject_image_into_polar(IM)
 
    if inverse:
        # pseudo inverse Abel transform of the polar image, removes background
        # to enhance transition peaks
        polarIM = abel.dasch.two_point_transform(polarIM.T).T

    # limit radial range of polar image, if selected
    radial = radial_coord[:, 0]
    if radial_range is not None:
        subr = np.logical_and(radial > radial_range[0]*zoom,
                              radial < radial_range[1]*zoom)
        polarIM = polarIM[subr]
        radial = radial[subr]

    # split image into n-slices
    slices = np.array_split(polarIM, nslices, axis=1)
    slice_angles_array = np.array_split(angle_coord[0], nslices)
    # Fix me! - np.hsplit does not cope with uneven split
    #           there must be a better way to determine mean slice angle?
    slice_angles = []
    for ang in slice_angles_array:
        slice_angles.append(np.mean(ang))
    slice_angles = np.array(slice_angles)

    # evaluate radial scaling factor for each slice
    radcorr = correction(slice_angles, slices, radial, method=method)

    # spline radial scaling vs angle
    radcorrspl = UnivariateSpline(slice_angles, radcorr, s=smooth, ext=3)

    # apply the correction
    IMcirc = circularize(IM, radcorrspl, ref_angle=ref_angle)

    if zoom > 1:
        # return to original image size
        IMcirc = ndimage.zoom(IMcirc, 1/zoom)

    if return_correction:
        return IMcirc, slice_angles, radcorr, radcorrspl
    else:
        return IMcirc


def circularize(IM, radcorrspl, ref_angle=None):
    """
    Remap image from its distorted grid to the true cartesian grid.

    Parameters
    ----------
    IM : numpy 2D array
        original image

    radcorrspl: numpy func(theta)
        spline function to evaluate radial correction at a given angle

    """
    # cartesian coordinate system
    Y, X = np.indices(IM.shape)

    row, col = IM.shape
    origin = (col//2, row//2)  # odd image 

    # coordinates relative to center
    X -= origin[0]
    Y = origin[1] - Y   # negative values below the axis
    theta = np.arctan2(X, Y)  # referenced to vertical direction

    # radial scale factor at angle = ref_angle
    if ref_angle == None:
        factor = np.mean(radcorrspl(theta))
    else:
        factor = radcorrspl(ref_angle)

    print("ref_angle = {}, factor = {:g}".format(ref_angle, factor))

    # radial correction
    Xactual = X*factor/radcorrspl(theta)
    Yactual = Y*factor/radcorrspl(theta)

    # @DanHickstein magic
    # https://github.com/PyAbel/PyAbel/issues/186#issuecomment-275471271
    IMcirc = ndimage.interpolation.map_coordinates(IM,
                                  (origin[1] - Yactual, Xactual + origin[0]))

    return IMcirc


def residual(param, radial, profile, previous):
    """ compare radially scaled intensity profile with adjacent 'previous'
        profile, to determine the radial scale factor param[0], and
        amplitude param[1].

    """
    newradial = radial*param[0]
    spline_prof = UnivariateSpline(newradial, profile, s=0, ext=3)
    newprof = spline_prof(radial)*param[1]

    # residual cf adjacent slice profile
    return newprof - previous


def correction(slice_angles, slices, radial, method):
    """ evaluate a radial scaling factor to align intensity profile features,
        between adjacent angular slices.

    """

    if method == "argmax":
        # follow position of intensity maximum
        pkpos = []

        for ang, aslice in zip(slice_angles, slices):
            profile = aslice.sum(axis=1)  # intensity vs radius for slice
            pkpos.append(profile.argmax())  # store index of peak position

        # radial scaling factor relative to peak max in first angular slice
        sf = radial[pkpos[0]]/radial[pkpos]

    elif method == "lsq":
        # least-squares radially scale intensity profile to match previous slice

        fitpar = np.array([1.0, 1.0])  # radial scale factor, amplitude
        sf = []
        sf.append(1)  # first slice nothing to compare with
        previous = slices[0].sum(axis=1)

        for ang, aslice in zip(slice_angles[1:], slices[1:]):
            profile = aslice.sum(axis=1)  # intensity vs radius for slice

            result = leastsq(residual, fitpar, args=(radial, profile, previous))

            sf.append(result[0][0])  # radial scale factor direct from lsq

            previous += residual(result[0], radial, profile, previous)
            fitpar = result[0]

    else:
        raise ValueError("method variable must be one of 'argmax' or 'lsq',"
                         " not '{}'".format(method))

    return sf
