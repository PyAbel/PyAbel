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
# Stephen Gibson - Australian National University, Australia
# Jason Gascooke - Flinders University, Australia
#
# This algorithm is adapted by Jason Gascooke from the article
#   E. W. Hansen and P-L. Law
#  "Recursive methods for computing the Abel transform and its inverse"
#   J. Opt. Soc. Am A2, 510-520 (1985) doi: 10.1364/JOSAA.2.000510
#
#  J. R. Gascooke PhD Thesis:
#   "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals
#    Molecule Dissociation", Flinders University, 2000.
#
# Implemented in Python, with image quadrant co-adding, by Steve Gibson
# 2018-03   : NB method applies to grid centered (even columns), not
#             pixel-centered (odd column) image see #206, #211
# 2018-02   : Drop one array dimension, use numpy broadcast multiplication
# 2015-12-16: Modified to calculate the forward Abel transform
# 2015-12-03: Vectorization and code improvements Dan Hickstein and
#             Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
#############################################################################


def hansenlaw_transform(IM, dr=1, direction='inverse', shift=-0.5, **kwargs):
    r"""Forward/Inverse Abel transformation using the algorithm of
    `Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985)
    <http://dx.doi.org/10.1364/JOSAA.2.000510>`_ equation 2a:


    .. math::

     f(r) = -\frac{1}{\pi} \int_{r}^{\infty} \frac{g^\prime(R)}{\sqrt{R^2-r^2}} dR,

    where

    :math:`f(r)` is the reconstructed image (source) function,
    :math:`g'(R)` is the derivative of the projection (measured) function

    The Hansen and Law approach treats the Abel transform as a system modeled
    by a set of linear differential equations, with :math:`f(r)` (forward) or
    :math:`g'(R)` (inverse) the driving function.

    Evaluation follows Eqs. (15 or 17), using (16a), (16b), and (16c or 18) of
    the Hansen and Law paper. For the full image transform, use the
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
    image at a time.


    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    dr : float
        sampling size (=1 for pixel images), used for Jacobian scaling

    direction : string ('forward' or 'inverse')
        ``forward`` or ``inverse`` Abel transform

    shift : float
        transform-pair better agreement if image shifted across
        `scipy.ndimage.shift(IM, (0, -shift))`.  Default `shift=-1/2` pixel

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
        whole image. 
    """

    # Hansen & Law parameters of exponential approximation, Table 1.
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    IM = np.atleast_2d(IM)

    # shift image across (default -1/2 pixel) gives better transform-pair
    IMS = interpolation.shift(IM, (0, shift))

    AIM = np.empty_like(IM)  # forward/inverse Abel transform image

    rows, N = IM.shape  # shape of input quadrant (half)
    K = h.size  # using H&L nomenclature

    # enumerate columns n = 0 is Rmax, the right side of image
    n = np.arange(N-1)  # n =  0, ..., N-2

    num = N - n
    denom = num - 1  # N-n-1 in Hansen & Law
    ratio = num/denom  # (N-n)/(N-n-1) = N/(N-1), ..., 4/3. 3/2, 2/1

    # phi array Eq (16a), diagonal array, for each pixel
    phi = np.empty((N-1, K))
    for k in range(K):
        phi[:, k] = ratio**lam[k]

    # Gamma array, Eq (16b), with gamma Eq (16c) forward, or Eq (18) inverse
    gamma = np.empty_like(phi)
    if direction == "forward":
        lam += 1
        for k in range(K):
            gamma[:, k] = h[k]*2*denom*(1 - ratio**lam[k])/lam[k]  # (16c)
        gamma *= -np.pi*dr  # Jacobian - saves scaling the transform later

        # driving function = raw image. Copy so input image not mangled
        drive = IMS

    else:  # gamma for inverse transform
        gamma[:, 0] = -h[0]*np.log(ratio)  # Eq. (18 lamda=0)
        for k in range(1, K):
            gamma[:, k] = h[k]*(1 - phi[:, k])/lam[k]  # Eq. (18 lamda<0)

        # driving function derivative of the image intensity profile
        drive = np.gradient(IMS, dr, axis=-1)

    # Hansen and Law Abel transform ---- Eq. (15) forward, or Eq. (17) inverse
    # transforms every image row during the column iteration
    x = np.zeros((K, rows))
    for nindx, pixelcol in zip(n, -n-1):  # 
        x  = phi[nindx][:, None]*x + gamma[nindx][:, None]*drive[:, pixelcol]
        AIM[:, pixelcol] = x.sum(axis=0)

    # missing 1st column
    AIM[:, 0] = AIM[:, 1]

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    return AIM
