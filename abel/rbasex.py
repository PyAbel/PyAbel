# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.linalg import inv

from abel.tools.vmi import Distributions
from abel.tools.symmetry import put_image_quadrants

###############################################################################
#
# rBasex — Abel transform for velocity-map images (~spherical harmonics),
#          similar to pBasex, but with analytical basis functions and without
#          pixel-grid transformation; a simplified version of the method
#          described in M. Ryazanov Ph.D. dissertation.
#
# pBasex:
#   Gustavo A. Garcia, Laurent Nahon, Ivan Powis,
#   “Two-dimensional charged particle image inversion using a polar basis
#    function expansion”,
#   Review of Scientific Instruments 75, 4989 (2004).
#   http://dx.doi.org/10.1063/1.1807578
#
# Ryazanov:
#   Mikhail Ryazanov,
#   “Development and implementation of methods for sliced velocity map imaging.
#    Studies of overtone-induced dissociation and isomerization dynamics of
#    hydroxymethyl radical (CH₂OH and CD₂OH)”,
#   Ph.D. dissertation, University of Southern California, 2012.
#   https://search.proquest.com/docview/1289069738
#   http://digitallibrary.usc.edu/cdm/ref/collection/p15799coll3/id/112619
#
###############################################################################


# Caches and their parameters
_prm = None  # [shape, origin, rmax, weights]
_dst = None  # Distribution object
_bs_prm = None  # [Rmax]
_bs = None  # [P0, P2] — projected functions
_trf = None  # [Af0, Af2] — forward transform matrices
_tri = None  # [Ai0, Ai2] — inverse transform matrices


def rbasex_transform(IM, origin='center', rmax='MIN',
                     weights=None, direction='inverse'):
    """
    This function takes the input image and outputs its forward or inverse Abel
    transform as an image and its radial distributions.

    The **origin**, **rmax** and **weights** parameters are passed to
    :class:`abel.tools.vmi.Distributions`, so see its documentation for their
    detailed descriptions.

    Parameters
    ----------
    IM : m × n numpy array
        the image to be transformed
    origin : tuple of int or str
        image origin, explicit in the (row, column) format, or as a location
        string
    rmax : int or string
        largest radius to include in the transform
    weights : m × n numpy array, optional
        weighting factors for each pixel. The array shape must match the image
        shape. Parts of the image can be excluded from analysis by assigning
        zero weights to their pixels.
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    Returns
    -------
    recon : 2D numpy array
        the transformed image. Is centered and might have different dimensions
        than the input image.
    distr : Distributions.Results object
        the object from which various distributions for the transformed image
        can be retrieved, see :class:`abel.tools.vmi.Distributions.Results`
    """

    p0, p2 = _profiles(IM, origin, rmax, weights)

    Rmax = _dst.rmax

    A0, A2 = get_bs_cached(Rmax, direction)

    c0 = A0.dot(p0)
    c2 = A2.dot(p2)

    distr = Distributions.Results(np.arange(Rmax + 1), np.array([c0, c2]),
                                  2, False)
    Q = _image(c0, c2)[::-1]
    height, width = _dst.Qheight, _dst.Qwidth
    recon = put_image_quadrants((Q, Q, Q, Q),
                                (2 * height - 1, 2 * width - 1))

    return recon, distr


def _profiles(IM, origin, rmax, weights):
    """
    Get radial profiles of cos^n theta terms from the input image.
    """
    # the Distributions object is cached to speed up further calculations,
    # plus its cos^n theta matrices are used later to construct the transformed
    # image
    global _prm, _dst

    prm = [IM.shape, origin, rmax, weights]
    if _prm != prm:
        _prm = prm
        _dst = Distributions(origin=origin, rmax=rmax, weights=weights,
                             use_sin=False, method='linear')

    return _dst(IM).cos()


def _image(c0, c2):
    """
    Create transformed image from its cos^n theta radial profiles.
    """
    rbin, wl, wu, cos2 = _dst.bin, _dst.wl, _dst.wu, _dst.c[1]

    c0l = np.append(c0, 0)
    c2l = np.append(c2, 0)
    c0u = np.append(c0[1:], [0, 0])
    c2u = np.append(c2[1:], [0, 0])

    return wl * (c0l[rbin] + c2l[rbin] * cos2) + \
           wu * (c0u[rbin] + c2u[rbin] * cos2)


def _bs_rbasex(Rmax):
    """
    Compute radial parts of basis projections for R and radii up to Rmax.
    """
    def psqrt(x):
        return np.sqrt(x, out=np.zeros_like(x), where=x > 0)

    def plog(x):
        return np.log(x, out=np.zeros_like(x), where=x > 0)

    def F1(n, r, z):
        if n == 0:
            return z
        elif n == 2:
            return r * np.arctan2(z, r)

    def Frho(n, r, z):
        rho = np.sqrt(r**2 + z**2)
        if n == 0:
            return (z * rho + r**2 * plog(z + rho)) / 2
        elif n == 2:
            return r**2 * plog(z + rho)

    def Z(R, r):
        return psqrt(R**2 - r**2)

    def p(R, n, r):
        ZR = Z(R, r)
        ZRm1 = Z(R - 1, r)
        ZRp1 = Z(R + 1, r)
        return 2 * (2 * (Frho(n, r, ZR) - R * F1(n, r, ZR)) +
                    (R - 1) * F1(n, r, ZRm1) - Frho(n, r, ZRm1) +
                    (R + 1) * F1(n, r, ZRp1) - Frho(n, r, ZRp1))

    r = np.arange(Rmax + 1, dtype=float)
    P0 = np.empty((Rmax + 1, Rmax + 1))
    P2 = np.empty((Rmax + 1, Rmax + 1))
    for R in range(Rmax + 1):
        P0[R] = p(R, 0, r)
        P2[R] = p(R, 2, r)

    return P0.T, P2.T


def get_bs_cached(Rmax, direction='inverse'):
    """
    Internal function.

    Gets the basis set (from cache or runs computations)
    and calculates the transform matrix.
    Loaded/calculated matrices are also cached in memory.

    Parameters
    ----------
    Rmax : int
        largest radius to be transformed
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    Returns
    -------
    A : tuple of 2D numpy arrays
        (**Rmax** + 1) × (**Rmax** + 1) matrices of the Abel transform (forward
        or inverse) for each angular order
    """
    global _bs_prm, _bs, _trf, _tri

    prm = [Rmax]
    if _bs is None or _bs_prm != prm:
        _bs_prm = prm
        _bs = _bs_rbasex(Rmax)
        _trf = None
        _tri = None

    if direction == 'forward':
        if _trf is None:
            _trf = _bs
        return _trf
    else:  # 'inverse'
        if _tri is None:
            A0, A2 = _bs
            A2[0, 0] = 1  # (to avoid degeneracy)
            Ai0 = inv(A0)
            Ai2 = inv(A2)
            _tri = (Ai0, Ai2)
        return _tri


def cache_cleanup():
    """
    Utility function.

    Frees the memory caches created by ``get_bs_cached()``.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global _prm, _dst, _bs_prm, _bs, _trf, _tri

    _prm = None
    _dst = None
    _bs_prm = None
    _bs = None
    _trf = None
    _tri = None
