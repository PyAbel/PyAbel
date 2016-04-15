# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from time import time
from math import exp, log, pow, pi

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
# 2015-12-03: Vectorization and code improvements Dan Hickstein and Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
#############################################################################


def hansenlaw_transform(IM, dr=1, direction="inverse"):
    r"""Forward/Inverse Abel transformation using the algorithm of
    `Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985) 
    <http://dx.doi.org/10.1364/JOSAA.2.000510>`_ equation 2a: 
    
    
    .. math:: f(r) = -\frac{1}{\pi} \int_{r}^{\infty} \frac{g^\prime(R)}{\sqrt{R^2-r^2}} dR,
    
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
        
    This function performs the Hansen-Law transform on only one "right-side" image, 
    typically one quadrant of the full image: ::

        Qtrans = abel.hansenlaw.hansenlaw_transform(Q, direction="inverse")

    Recursion method proceeds from the outer edge of the image
    toward the image centre (origin). i.e. when ``n=N-1``, ``R=Rmax``, and
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
        Use ``abel.tools.center.center_image(IM, method='com', odd_size=True)`` 
    """

    IM = np.atleast_2d(IM)
    N = np.shape(IM)         # shape of input quadrant (half)
    AIM = np.zeros(N)        # forward/inverse Abel transform image

    rows, cols = N

    # Two alternative Gamma functions for forward/inverse transform
    # Eq. (16c) used for the forward transform
    def fgamma(Nm, lam, n):
        return 2*n*(1-np.power(Nm, lam+1))/(lam+1)

    # Eq. (18) used for the inverse transform
    def igammalt(Nm, lam, n):
        return (1-np.power(Nm, lam))/(pi*lam)

    def igammagt(Nm, lam, n):
        return -np.log(Nm)/pi

    if direction == "inverse":   # inverse transform
        gammagt = igammagt   # special case lam = 0.0
        gammalt = igammalt   # lam < 0.0

        # g' - derivative of the intensity profile
        if rows > 1:
            gp = np.gradient(IM)[1]
            # second element is gradient along the columns
        else:  # If there is only one row
            gp = np.atleast_2d(np.gradient(IM[0]))

    else:  # forward transform
        gammagt = gammalt = fgamma
        gp = IM

    # ------ The Hansen and Law algorithm ------------
    # iterate along columns, starting outer edge (right side)
    # toward the image center

    # constants listed in Table 1.
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    K = np.size(h)
    nn = np.arange(cols-2, 0, -1, dtype=int)

    Nm = (nn+1)/nn          # R0/R

    X = np.zeros((K, rows))
    Gamma = np.zeros((cols-2, K, 1))
    Phi = np.zeros((cols-2, K, K))

    # Gamma_n and Phi_n  Eq. (16a) and (16b)
    # lam = 0.0
    Gamma[:, 0, 0] = h[0]*gammagt(Nm, lam[0], nn)   
    Phi[:, 0, 0] = 1
    # lam < 0.0
    for k in range(1, K): 
        Gamma[:, k, 0] = h[k]*gammalt(Nm, lam[k], nn)  
        Phi[:, k, k] = np.power(Nm, lam[k])   # diagonal matrix Eq. (16a)

    # Eq. (15) forward, or (17) inverse
    for i, n in enumerate(nn):
        X = np.dot(Phi[i], X) + Gamma[i]*gp[:, n]
        AIM[:, n] = X.sum(axis=0)

    # center pixel column
    AIM[:, 0] = AIM[:, 1]

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    if direction == "inverse":
        return AIM*np.pi/dr    # 1/dr - from derivative
    else:
        return -AIM*np.pi*dr   # forward still needs '-' sign
