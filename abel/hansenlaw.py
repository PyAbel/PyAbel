# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from time import time
from math import exp, log, pow, pi

"""
hansenlaw - a recursive method forward/inverse Abel transform algorithm

Stephen Gibson - Australian National University, Australia
Jason Gascooke - Flinders University, Australia

This algorithm is adapted by Jason Gascooke from the article
  E. W. Hansen and P-L. Law
 "Recursive methods for computing the Abel transform and its inverse"
  J. Opt. Soc. Am A2, 510-520 (1985) doi: 10.1364/JOSAA.2.000510

 J. R. Gascooke PhD Thesis:
  "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals
   Molecule Dissociation", Flinders University, 2000.

Implemented in Python, with image quadrant co-adding, by Steve Gibson
2015-12-16: Modified to calculate the forward Abel transform
2015-12-03: Vectorization and code improvements Dan Hickstein and Roman Yurchak
            Previously the algorithm iterated over the rows of the image
            now all of the rows are calculated simultaneously, which provides
            the same result, but speeds up processing considerably.
"""

_hansenlaw_header_docstring = \
    """
    Forward/Inverse Abel transformation using the algorithm of:
    Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985).::


                   ∞
                   ⌠
               -1  ⎮   g'(R)
       f(r) =  ─── ⎮ ──────────── dR      Eq. (2a)
                π  ⎮    _________
                   ⎮   ╱  2    2
                   ⎮ ╲╱  R  - r
                   ⌡
                   r


    f(r)
        is reconstructed image (source) function
    g'(R)
        is derivative of the projection (measured) function

    Evaluation via Eq. (15 or 17), using (16a), (16b), and (16c or 18)

    f = iabel_hansenlaw(g)
        inverse Abel transform of image g
    g = fabel_hansenlaw(f)
        forward Abel transform of image f
    (f/i)abel_hansenlaw_transform()
        core algorithm
    """

_hansenlaw_transform_docstring = \
    """

    Core Hansen and Law Abel transform

    Recursion method proceeds from the outer edge of the image
    toward the image centre (origin). i.e. when n=N-1, R=Rmax, and
    when n=0, R=0. This fits well with processing the image one
    quadrant (chosen orientation to be rightside-top), or one right-half
    image at a time.

    Use (f/i)abel_transform (IM) to transform a whole image

    Parameters
    ----------
    IM : 2D np.array
        One quadrant (or half) of the image oriented top-right::

             +--------      +--------+              |
             |      *       | *      |              |
             |   *          |    *   |  <----------/
             |  *           |     *  |
             +--------      o--------+
             |  *           |     *  |
             |   *          |    *   |
             |     *        | *      |
             +--------      +--------+

             Image centre `o' should be within a pixel
             (i.e. an odd number of columns)
             [Use abel.tools.center.find_image_center_by_slice () to transform] 

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling

    direction : string
        'forward' or 'inverse' Abel transform

    Returns
    -------
    AIM : 2D np.array
        forward/inverse Abel transform image

    """


def fabel_hansenlaw(IM, dr=1):
    """
    Forward Abel transform for one-quadrant
    """
    return hansenlaw_transform(IM, dr=dr, direction="forward")


def iabel_hansenlaw(IM, dr=1):
    """
    Inverse Abel transform for one-quadrant
    """
    return hansenlaw_transform(IM, dr=dr, direction="inverse")


def hansenlaw_transform(IM, dr=1, direction="inverse"):
    """
    Hansen and Law JOSA A2 510 (1985) forward and inverse Abel transform
    for right half (or right-top quadrant) of an image.
    """

    IM = np.atleast_2d(IM)
    N = np.shape(IM)         # shape of input quadrant (half)
    AIM = np.zeros(N)        # forward/inverse Abel transform image

    rows, cols = N

    # constants listed in Table 1.
    h = [0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3]
    lam = [0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1]

    K = np.size(h)
    X = np.zeros((rows, K))

    # Two alternative Gamma functions for forward/inverse transform
    # Eq. (16c) used for the forward transform
    def fgamma(Nm, lam, n):
        return 2*n*(1-pow(Nm, (lam+1)))/(lam+1)

    # Eq. (18) used for the inverse transform
    def igamma(Nm, lam, n):
        if lam < -1:
            return (1-pow(Nm, lam))/(pi*lam)
        else:
            return -np.log(Nm)/pi

    if direction == "inverse":   # inverse transform
        gamma = igamma
        # g' - derivative of the intensity profile
        if rows > 1:
            gp = np.gradient(IM)[1]
            # second element is gradient along the columns
        else:  # If there is only one row
            gp = np.atleast_2d(np.gradient(IM[0]))
    else:  # forward transform
        gamma = fgamma
        gp = IM

    # ------ The Hansen and Law algorithm ------------
    # iterate along columns, starting outer edge (right side)
    # toward the image center

    for n in range(cols-2, 0, -1):
        Nm = (n+1)/n          # R0/R

        for k in range(K):  # Iterate over k, the eigenvectors?
            X[:, k] = pow(Nm, lam[k])*X[:, k] +\
                     h[k]*gamma(Nm, lam[k], n)*gp[:, n]  # Eq. (15 or 17)
        AIM[:, n] = X.sum(axis=1)

    # special case for the end pixel
    AIM[:, 0] = AIM[:, 1]

    #for some reason shift by 1 pixel aligns better? - FIX ME!
    if direction == "inverse":
        AIM = np.c_[AIM[:, 1:],AIM[:, -1]]

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    if direction == "inverse":
        return AIM*np.pi/dr    # 1/dr - from derivative
    else:
        return -AIM*np.pi*dr   # forward still needs '-' sign

# append the same docstring to all functions - borrowed from @rth
iabel_hansenlaw.__doc__ += _hansenlaw_header_docstring + _hansenlaw_transform_docstring
fabel_hansenlaw.__doc__ += _hansenlaw_header_docstring + _hansenlaw_transform_docstring
hansenlaw_transform.__doc__ += _hansenlaw_header_docstring + _hansenlaw_transform_docstring
