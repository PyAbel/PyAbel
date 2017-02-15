# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from scipy import ndimage
from scipy.interpolate import UnivariateSpline
from scipy.optimize import leastsq

import abel

# see https://github.com/PyAbel/PyAbel/issues/186 for discussion


def circularize_image(IM, method="lsq", center=None, radial_range=None,
                      dr=0.5, dt=0.5, smooth=0, ref_angle=None, 
                      inverse=False, return_correction=False):
    """
    This function corrects for radial distortion in an image that consists of
    features that exist at a constant radius. In other words, the image must
    consist of "circle-like" structures. This function attempts to correct for
    any distortion and ensures that the circle-like structures are perfect
    circles.

    This function is especially useful for correcting the image obtained with
    a velocity-map-imaging spectrometer with imperfect electro-static lenses,
    allowing a high-resolution 1D photoelectron distribution to be extracted.

    The algorithm is to split the image into "slices" at many different angles
    (set by `nslices`) and compare adjacent slices. A scaling factor is found
    which aligns each slice with the previous slice. The image is then
    corrected using a spline function that smoothly connects the discrete
    scaling factors as a continuous function of angle.

    The circularization algorithm should only be applied to a well-centered
    image.


    Parameters
    ----------
    IM : numpy 2D array

    method : str
        method to align slice profiles,
        "argmax" - compare intensity-profile.argmax() of each radial slice.
        "lsq" - determine radial scaling factor by minimizing the
                intensity-prfoile with an adjacent slice intensity-profile.

    center : str, float tuple, or None
        pre-center image using :func:`abel.tools.center.center_image`
        "com", "convolution", "gaussian", "image_center", "slice", or
        float tuple center (y, x).

    radial_range : tuple, or None
        limit slice comparison to the radial range tuple (rmin, rmax), in
        pixels, from the image center. Used to determine the alignment
        around particular peaks. 

    dr : float
        Radial grid size for the polar coordinate image, default = 0.5 pixel.
        Small values may improve the distortion correction, which is often of
        sub-pixel dimensions, at the cost of reduced signal to noise for the
        slice intensity profile. 
        See also :func:`abel.tools.polar.reproject_image_into_polar`

    dt : float
        Angular grid size.
        See also :func:`abel.tools.polar.reproject_image_into_polar`

    smooth : float
        This value is passed to the `scipy.interpolate.UnivariateSpline`
        function and controls how smooth the spline interpolation is. A value 
        of zero corresponds to a spline that runs through all of the points,
        and higher values correspond to a smoother spline function. 
        It is important to examine the relative peak position (scaling factor)
        data and how well it is represented by the spline function. Use the
        option `return_correction=True` to examine this data. Typically,
        `smooth` may remain zero, noisy data may require some smoothing.

    ref_angle : None or float
        reference angle for which radial coordinate is unchanged.
        Angle varies between -pi to pi, with zero angle vertical.
        None uses numpy.mean(radial scale factors), which attempts to maintain
        the same average radial scaling. This approximation is likely valid,
        unless you know for certain that a specific angle of your image
        corresponds to an undistorted image."

    inverse : bool
        inverse Abel transform the *polar* image, to remove the background
        intensity. This may improve the signal to noise, allowing weak 
        intensity featured to be followed in angle. 

        Note that this step is only for the purposes of allowing the algorithm
        to better follow peaks in the image. It does not affect the final 
        image that is returned, expect for (hopefully) slightly improving the
        precision of the distortion correction.

    return_correction : bool
        additional outputs, see below

    Returns
    -------
    IMcirc : numpy 2D array, same size as input
        Circularized imput image

    (if return_correction is True)
    slice_angles : numpy 1D array
        Mid-point angle (radians) of image slice

    radial_corrections : numpy 1D array
        radial correction scale factors, at any angle

    function : numpy function that accepts numpy.array
        function used to evaluate the radial correction at each angle

    """

    if center is not None:
        # convenience function for the case image is not centered
        IM = abel.tools.center.center_image(IM, center=center)

    # map image into polar coordinates - much easier to slice
    # cartesian (Y, X) -> polar (Radius, Theta)
    polarIM, radial_coord, angle_coord =\
             abel.tools.polar.reproject_image_into_polar(IM, dr=dr, dt=dt)
 
    if inverse:
        # pseudo inverse Abel transform of the polar image, removes background
        # to enhance transition peaks
        polarIM = abel.dasch.two_point_transform(polarIM.T).T

    # 1-D coordinate arrays
    angles = angle_coord[0]  # angle coordinate
    radial = radial_coord[:, 0]  # radial coordinate

    # limit radial range of polar image, if selected
    if radial_range is not None:
        subr = np.logical_and(radial > radial_range[0]*zoom,
                              radial < radial_range[1]*zoom)
        polarIM = polarIM[subr]
        radial = radial[subr]

    # evaluate radial scaling factor for each slice
    radcorr = correction(polarIM.T, radial, angles, method=method)

    # spline radial scaling vs angle
    radcorrspl = UnivariateSpline(angles, radcorr, s=smooth, ext=3)

    # apply the correction
    IMcirc = circularize(IM, radcorrspl, ref_angle=ref_angle)

    if return_correction:
        return IMcirc, angles, radcorr, radcorrspl
    else:
        return IMcirc


def circularize(IM, radcorrspl, ref_angle=None):
    """
    Remap image from its distorted grid to the true cartesian grid.

    Parameters
    ----------
    IM : numpy 2D array
        original image

    radcorrspl : numpy function that accepts numpy.array of angles
        function to evaluate radial correction at a given angle

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

    # radial correction
    Xactual = X*factor/radcorrspl(theta)
    Yactual = Y*factor/radcorrspl(theta)

    # @DanHickstein magic
    # https://github.com/PyAbel/PyAbel/issues/186#issuecomment-275471271
    IMcirc = ndimage.interpolation.map_coordinates(IM,
                                  (origin[1] - Yactual, Xactual + origin[0]))

    return IMcirc


def _residual(param, radial, profile, previous):
    """ compare radially scaled intensity profile with adjacent 'previous'
        profile, to determine the radial scale factor param[0], and
        amplitude param[1].

    """
    radial_scaling, amplitude = param[0], param[1]

    newradial = radial*radial_scaling
    spline_prof = UnivariateSpline(newradial, profile, s=0, ext=3)
    newprof = spline_prof(radial)*amplitude

    # residual cf adjacent slice profile
    return newprof - previous


def correction(polarIMTrans, radial, angles, method):
    """ `leastsq` residuals function.

    Evaluate the difference between a radial-scaled intensity profile
    and its adjacent angular slice.


    Parameters
    ----------
    polarIMTrans : numpy 2D array
        polar coordinate image, transposed so that each row is one angle

    radial : numpy 1D array
        radial coordinates of one column of polarIMTrans

    angles : numpy 1D array
        angle coordinates of one row of polarIMTrans

    method : str
        'lsq' : determine a radial correction factor that will align a
        radial intensity profile with the previous, adjacent slice.
        'argmax': radial correction factor from position of maximum intensity

    """

    if method == "argmax":
        # follow position of intensity maximum
        pkpos = []

        for ang, aslice in zip(angles, polarIMTrans):
            profile = aslice
            pkpos.append(profile.argmax())  # store index of peak position

        # radial scaling factor relative to peak max in first angular slice
        sf = radial[pkpos[0]]/radial[pkpos]

    elif method == "lsq":
        # least-squares radially scale intensity profile to match previous slice

        fitpar = np.array([1.0, 1.0])  # radial scale factor, amplitude
        sf = []
        sf.append(1)  # first slice nothing to compare with
        previous = polarIMTrans[0]

        for ang, aslice in zip(angles[1:], polarIMTrans[1:]):
            profile = aslice

            result = leastsq(_residual, fitpar, args=(radial, profile,
                             previous))

            sf.append(result[0][0])  # radial scale factor direct from lsq

            previous += _residual(result[0], radial, profile, previous)
            fitpar = result[0]

    else:
        raise ValueError("method variable must be one of 'argmax' or 'lsq',"
                         " not '{}'".format(method))

    return sf
