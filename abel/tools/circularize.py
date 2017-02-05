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
                      zoom=1, smooth=1.0e-7, nslices=32, inverse=False,
                      return_correction=False):
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
        IM = ndimage.zoom(IM, zoom)

    if center is not None:
        # convenience function in case image is not centered
        IM = abel.tools.center.center_image(IM, center=center)

    # map image into polar coordinates - much easier to slice
    # cartesian (Y, X) -> polar (Radius, Theta)
    polarIM, radial_coord, angle_coord =\
                           abel.tools.polar.reproject_image_into_polar(IM)

    if inverse:
        polarIM = abel.Transform(polarIM.T).transform.T

    # limit radial range of polar image, if selected
    radial = radial_coord[:, 0]
    if radial_range is not None:
        subr = np.logical_and(radial > radial_range[0]*zoom,
                              radial < radial_range[1]*zoom)
        polarIM = polarIM[subr]
        radial = radial[subr]

    # split image into n-slices
    slices = np.array_split(polarIM, nslices, axis=1)
    slice_angles = np.array(np.hsplit(angle_coord[0], nslices)).mean(axis=1)

    # evaluate radial scaling factor for each slice
    radcorr = correction(slice_angles, slices, radial, method=method)

    # spline radial scaling vs angle
    radcorrspl = UnivariateSpline(slice_angles, radcorr, s=smooth, ext=3)

    # apply the correction
    IMcirc = circularize(IM, radcorrspl)

    if zoom > 1:
        # return to original image size
        IMcirc = ndimage.zoom(IMcirc, 1/zoom)

    if return_correction:
        return IMcirc, slice_angles, radcorr, radcorrspl
    else:
        return IMcirc


def circularize(IM, radcorrspl):
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
    origin = (col//2 + col % 2, row//2 + row % 2)  # % handles odd size image

    # coordinates relative to center
    X -= origin[0]
    Y -= origin[1]
    theta = np.arctan2(Y, X)

    # radial correction
    Xactual = X/radcorrspl(theta)
    Yactual = Y/radcorrspl(theta)

    # @DanHickstein magic
    # https://github.com/PyAbel/PyAbel/issues/186#issuecomment-275471271
    IMcirc = ndimage.interpolation.map_coordinates(IM,
                                  (Yactual+origin[1], Xactual+origin[0]))

    return IMcirc


def residual(param, radial, profile, previous):

    newradial = radial*param[0]
    spline_prof = UnivariateSpline(newradial, profile, s=0, ext=3)
    newprof = spline_prof(radial)*param[1]

    # residual cf adjacent slice profile
    return newprof - previous


def correction(slice_angles, slices, radial, method):
    pkpos = []
    fitpar = np.array([1.0, 1.0])  # radial scale factor, amplitude
    for ang, aslice in zip(slice_angles, slices):
        profile = aslice.sum(axis=1)  # intensity vs radius for a given slice

        if method == "argmax":
            pkpos.append(profile.argmax())  # store index of peak position

        elif method == "lsq":
            if ang > slice_angles[0]:
                result = leastsq(residual, fitpar, args=(radial, profile,
                                                      previous))
                sf.append(result[0][0]) # radial scale factor direct from lsq
                profile = residual(result[0], radial, profile, previous) +\
                                    previous
            else:
                # first profile has nothing to compare with
                sf = []
                rsf = 1
                sf.append(1)

            previous = profile

    if method == "argmax":
        # radial scaling factor referenced to the position of the first
        # angular slice
        sf = radial[pkpos]/radial[pkpos[0]]

    elif method == "lsq":
        sf[0] = sf[-1]

    return sf
