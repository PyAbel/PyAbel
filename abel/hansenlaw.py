# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

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
# 2015-12-16: Modified to calculate the forward Abel transform
# 2015-12-03: Vectorization and code improvements Dan Hickstein and
#             Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
#############################################################################


def hansenlaw_transform(IM, dr=1, direction='inverse', **kwargs):
    r"""Forward/Inverse Abel transformation using the algorithm of
    `Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985)
    <http://dx.doi.org/10.1364/JOSAA.2.000510>`_ equation 2a:


    .. math::

     f(r) = -\frac{1}{\pi} \int_{r}^{\infty} \frac{g^\prime(R)}{\sqrt{R^2-r^2}} dR,

    where

    :math:`f(r)` is the reconstructed image (source) function,
    :math:`g'(R)` is the derivative of the projection (measured) function

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

        Image centre ``o`` should be within a pixel (i.e. an odd number of columns)
        Use ``abel.tools.center.center_image(IM, center='com')``
    """

    IM = np.atleast_2d(IM)
    AIM = np.zeros_like(IM)  # forward/inverse Abel transform image

    # Hansen & Law constants as listed in Table 1.
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    rows, cols = np.shape(IM)  # shape of input quadrant (half)

    # enumerate columns n=0 is Rmax, right side of image
    n = np.arange(cols-2, -1, -1)  # n =  cols-2, ..., 0
    denom = cols - n - 1  #  N-n-1 in Hansen & Law
    ratio = (cols-n)/denom  #  (N-n)/(N-n-1) in Hansen & Law

    K = np.size(h)
    Phi = np.zeros((cols-1, K, K))
    Phi[:, 0, 0] = 1
    for k in range(1, K):
        Phi[:, k, k] = ratio**lam[k]   # diagonal matrix Eq. (16a)

    Gamma = np.zeros((cols-1, K, 1))
    if direction == "forward":  # forward transform
        lam1 = lam + 1
        for k in range(K):
            Gamma[:, k, 0] = h[k]*2*denom*(1 - ratio**lam1[k])/lam1[k]  # (16c)
        Gamma *= -np.pi*dr  # Jacobian - saves scaling the transform later

        gp = IM  # raw image

    else:  # inverse transform
        # special case for lamda = 0
        Gamma[:, 0, 0] = -h[0]*np.log(ratio)  # Eq. (18 lamda=0)

        # lam < 0
        for k in range(1, K):
            Gamma[:, k, 0] = h[k]*(1 - Phi[:, k, k])/lam[k]  # Eq. (18)
        Gamma /= dr  # Jacobian

        # g' - derivative of the intensity profile
        gp = np.gradient(IM, axis=-1)

    # -1/2 pixel shift of source (image, or derivative) better aligns transform 
    # see issue #206
    gp = (gp[:, 1:] + gp[:, :-1])/2

    # Hansen and Law Abel transform ---- Eq. (15) forward, or Eq. (17) inverse
    X = np.zeros((K, rows))
    for col in n:  # right image edge to left edge
        X = np.dot(Phi[col], X) + Gamma[col]*gp[:, col]
        AIM[:, col] = X.sum(axis=0)

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    return AIM
