# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import interpolation

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


def hansenlaw_transform(im, dr=1, direction='inverse', hold_order=1, 
                        sub_pixel_shift=-0.35, **kwargs):
    r"""Forward/Inverse Abel transformation using the algorithm of:

    `E. W. Hansen "Fast Hankel Transform" IEEE Trans. Acoust. Speech Signal
    Proc. 33, 666 (1985) <https://dx.doi.org/10.1109/TASSP.1985.1164579>`_

    and

    `E. W. Hansen and P.-L. Law
    "Recursive methods for computing the Abel transform and its inverse"
    J. Opt. Soc. Am. A 2, 510-520 (1985)
    <https://dx.doi.org/10.1364/JOSAA.2.000510>`_ 

    For the full image transform, use the
    class :class:``abel.Transform``.

    For the inverse Abel transform of image g: ::

      f = abel.Transform(g, direction="inverse", method="hansenlaw").transform

    For the forward Abel transform of image f: ::

      g = abel.Transform(r, direction="forward", method="hansenlaw").transform

    This function performs the Hansen-Law transform on only one "right-side"
    image, typically one quadrant of the full image: ::

        Qtrans = abel.hansenlaw.hansenlaw_transform(Q, direction="inverse")

    Recursion method proceeds from the outer edge of the image
    toward the image centre (origin). i.e. when ``n=cols-1``, ``R=Rmax``, and
    when ``n=0``, ``R=0``. This fits well with processing the image one
    quadrant (chosen orientation to be rightside-top), or one right-half
    image at a time.  For the full image transform, use the class 
    :class:``abel.Transform``.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        Right-side half-image (or quadrant). See figure below.

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling.

    direction : string 'forward' or 'inverse'
        ``forward`` or ``inverse`` Abel transform.

    hold_order : int 0 or 1
        First- or zero-order hold approximation used in the evaluation of
        state equation integral.  `1` (default) yields a more accurate
        transform. `0` gives the same result as the original implementation
        of the `hansenlaw` method. 

    sub_pixel_shift : float -0.35 default
        For the zero-order hold approximation `hold_order=0` a sub-pixel 
        left-shift of the driving function (image-forward or gradient-inverse)
        improves the transform alignment with the other PyAbel methods,
        and Abel transform-pair functions.  See the discussion in PR #211.

    Returns
    -------
    AIM : 1D or 2D numpy array
        forward/inverse Abel transform half-image


    .. note::  Image should be a right-side image, like this: ::

        .         +--------      +--------+
        .         |      *       | *      |
        .         |   *          |    *   |  <---------- IM
        .         |  *           |     *  |
        .         +--------      o--------+
        .         |  *           |     *  |
        .         |   *          |    *   |
        .         |     *        | *      |
        .         +--------      +--------+

        In accordance with all PyAbel methods the image center ``o`` is
        defined to be mix-pixel i.e. an odd number of columns, for the
        full image.
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

    im = np.atleast_2d(im)   # 2D input image
    aim = np.empty_like(im)  # Abel transform array
    rows, cols = im.shape

    if direction == 'forward':
        drive = -2*dr*np.pi*im.copy()  # include Jacobian
        a = 1  # integration increases lambda + 1
    else:  # inverse Abel transform
        drive = np.gradient(im, dr, axis=-1)
        a = 0  # due to 1/piR factor

    n = np.arange(cols-1, 1, -1)

    phi = np.empty((n.size, h.size))
    for k, lamk in enumerate(lam):
        phi[:, k] = (n/(n-1))**lamk

    gamma0 = I(n, lam, a)*h

    if hold_order == 0:  # Hansen (& Law) zero-order hold approximation
        B1 = gamma0
        B0 = gamma0*0  # empty array
        # sub-pixel left shift improves transform alignment, see PR #211
        drive = interpolation.shift(drive, (0, -0.35))

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

    # missing columns at each side
    aim[:, 0] = aim[:, 1]
    aim[:, -1] = aim[:, -2]

    if rows == 1:
        aim = aim[0]  # flatten to a vector

    return aim
