# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

#############################################################################
# hansenlaw - a recursive method forward/inverse Abel transform algorithm
#
# Adapted from (see also PR #211):
#  [1] E. W. Hansen "Fast Hankel Transform"
#      IEEE Trans. Acoust. Speech, Signal Proc. 33(3), 666-671 (1985)
#      doi: 10.1109/TASSP.1985.1164579
#
# and:
#
#  [2] E. W. Hansen and P-L. Law
#      "Recursive methods for computing the Abel transform and its inverse"
#      J. Opt. Soc. Am A2, 510-520 (1985)
#      doi: 10.1364/JOSAA.2.000510
#
# 2019-04   : replace gradient with first-order finite difference as
#             per @MikhailRyazanov suggestion
# 2018-04   : New code rewrite, implementing the 1st-order hold approx. of
#             Ref. [1], with the assistance of Eric Hansen. See PR #211.
#
#             Original hansenlaw code was based on Ref. [2]
#
# 2018-03   : NB method applies to grid centered (even columns), not
#             pixel-centered (odd column) image see #206, #211
#             Apply, -1/2 pixel shift for odd column full image
# 2018-02   : Drop one array dimension, use numpy broadcast multiplication
# 2015-12-16: Modified to calculate the forward Abel transform
# 2015-12-03: Vectorization and code improvements Dan Hickstein and
#             Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
#
# Historically, this algorithm was adapted by Jason Gascooke from ref. [2] in:
#
#  J. R. Gascooke PhD Thesis:
#   "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals
#    Molecule Dissociation", Flinders University, 2000.
#
# Implemented in Python, with image quadrant co-adding, by Stephen Gibson (ANU)
# Significant code/speed improvements due to Dan Hickstein and Roman Yurchak
#
# Stephen Gibson - Australian National University, Australia
#
#############################################################################


def hansenlaw_transform(image, dr=1, direction='inverse', hold_order=0,
                        **kwargs):
    r"""Forward/Inverse Abel transformation using the algorithm of:

    `E. W. Hansen "Fast Hankel Transform" IEEE Trans. Acoust. Speech Signal
    Proc. 33, 666 (1985) <https://dx.doi.org/10.1109/TASSP.1985.1164579>`_

    and

    `E. W. Hansen and P.-L. Law
    "Recursive methods for computing the Abel transform and its inverse"
    J. Opt. Soc. Am. A 2, 510-520 (1985)
    <https://dx.doi.org/10.1364/JOSAA.2.000510>`_

    This function performs the Hansen-Law transform on only one "right-side"
    image: ::

        Abeltrans = abel.hansenlaw.hansenlaw_transform(image, direction='inverse')

    .. note::  Image should be a right-side image, like this: ::

        .         +--------      +--------+
        .         |      *       | *      |
        .         |   *          |    *   |  <---------- im
        .         |  *           |     *  |
        .         +--------      o--------+
        .         |  *           |     *  |
        .         |   *          |    *   |
        .         |     *        | *      |
        .         +--------      +--------+

        In accordance with all PyAbel methods the image center ``o`` is
        defined to be mid-pixel i.e. an odd number of columns, for the
        full image.


    For the full image transform, use the :class:``abel.Transform``.

    Inverse Abel transform: ::

      iAbel = abel.Transform(image, method='hansenlaw').transform

    Forward Abel transform: ::

      fAbel = abel.Transform(image, direction='forward', method='hansenlaw').transform


    Parameters
    ----------
    image : 1D or 2D numpy array
        Right-side half-image (or quadrant). See figure below.

    dr : float
        Sampling size, used for Jacobian scaling.
        Default: `1` (appliable for pixel images).

    direction : string 'forward' or 'inverse'
        ``forward`` or ``inverse`` Abel transform.
        Default: 'inverse'.

    hold_order : int 0 or 1
        The order of the hold approximation, used to evaluate the state equation
        integral. 
        `0` assumes a constant intensity across a pixel (between grid points)
        for the driving function (the image gradient for the inverse transform,
        or the original image, for the forward transform).
        `1` assumes a linear intensity variation between grid points, which may
        yield a more accurate transform for some functions (see PR 211).
        Default: `0`.

    Returns
    -------
    aim : 1D or 2D numpy array
        forward/inverse Abel transform half-image


    """

    # state equation integral_r0^r (epsilon/r)^(lamda+a) d\epsilon
    def I(n, lam, a):
        integral = np.empty((n.size, lam.size))

        ratio = n/(n-1)
        if a == 0:
            integral[:, 0] = -np.log(ratio)  # special case, lam=0

        ra = (n-1)**a
        k0 = not a  # 0 or 1

        for k, lamk in enumerate((lam+a)[k0:], start=k0):
            integral[:, k] = ra*(1 - ratio**lamk)/lamk

        return integral

    # parameters for Abel transform system model, Table 1.
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    image = np.atleast_2d(image)   # 2D input image
    aim = np.empty_like(image)  # Abel transform array
    rows, cols = image.shape

    if direction == 'forward':
        # copy image for the driving function, include Jacobian factor,
        drive = -2*dr*np.pi*np.copy(image)
        a = 1  # integration increases lambda + 1
    else:  # inverse Abel transform
        if hold_order == 0:
            # better suits sharp structure - see issue #249
            drive = np.zeros_like(image)
            drive[:, :-1] = (image[:, 1:] - image[:, :-1])/dr
        else:
            # hold_order=1 prefers gradient
            drive = np.gradient(image, dr, axis=-1)
        a = 0  # due to 1/piR factor

    n = np.arange(cols-1, 1, -1)

    phi = np.empty((n.size, h.size))
    for k, lamk in enumerate(lam):
        phi[:, k] = (n/(n-1))**lamk

    gamma0 = I(n, lam, a)*h

    if hold_order == 0:  # Hansen (& Law) zero-order hold approximation
        B1 = gamma0
        B0 = gamma0*0  # empty array

    else:  # Hansen first-order hold approximation
        gamma1 = I(n, lam, a+1)*h

        B0 = gamma1 - gamma0*(n-1)[:, None]  # f_n
        B1 = gamma0*n[:, None] - gamma1  # f_n-1

    # Hansen Abel transform  --------------------
    x = np.zeros((h.size, rows))

    for indx, col in enumerate(n-1):
        x = phi[indx][:, None]*x + B0[indx][:, None]*drive[:, col+1]\
                                 + B1[indx][:, None]*drive[:, col]
        aim[:, col] = x.sum(axis=0)

    # missing column at each side of image
    aim[:, 0] = aim[:, 1]
    aim[:, -1] = aim[:, -2]

    if rows == 1:
        aim = aim[0]  # flatten to a vector

    return aim
