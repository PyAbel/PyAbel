# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.linalg import inv, solve_triangular, svd

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
_prm = None  # [shape, origin, rmax, order, odd, weights]
_dst = None  # Distributions object
_bs_prm = None  # [Rmax, order, odd]
_bs = None  # [P[n]] — projected functions
_ibs = None  # [rbin, wl, wu, cos^n] — arrays for image construction
_trf = None  # [Af[n]] — forward transform matrices
_tri_prm = None  # [reg] — regularization parameters
_tri = None  # [Ai[n]] — inverse transform matrices


def rbasex_transform(IM, origin='center', rmax='MIN', order=2, odd=False,
                     weights=None, direction='inverse', reg=None, out='same'):
    r"""
    This function takes the input image and outputs its forward or inverse Abel
    transform as an image and its radial distributions.

    The **origin**, **rmax**, **order**, **odd** and **weights** parameters are
    passed to :class:`abel.tools.vmi.Distributions`, so see its documentation
    for their detailed descriptions.

    Parameters
    ----------
    IM : m × n numpy array
        the image to be transformed
    origin : tuple of int or str
        image origin, explicit in the (row, column) format, or as a location
        string
    rmax : int or string
        largest radius to include in the transform
    order : int
        highest order in the angular distributions, ≥ 0
    odd : bool
        include odd angular orders (enabled automatically if **order** is odd)
    weights : m × n numpy array, optional
        weighting factors for each pixel. The array shape must match the image
        shape. Parts of the image can be excluded from analysis by assigning
        zero weights to their pixels.
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed
    reg : None or tuple (str, float), optional
        regularization parameters for inverse Abel transform. ``None`` means no
        regularization, otherwise use a tuple (`type`, `strength`). Available
        `type`\ s are:

        ``'L2'``:
            Tikhonov :math:`L_2` regularization with `strength` as the square
            of the Tikhonov factor. This is the same as “Tikhonov
            regularization” used in BASEX, with almost identical effects on the
            radial distributions.
        ``'diff'``:
            Tikhonov regularization with the difference operator (approximation
            of the derivative) multiplied by the square root of `strength` as
            the Tikhonov matrix. This tends to produce less blurring, but more
            negative overshoots than ``'L2'``.
        ``'SVD'``:
            truncated SVD (singular value decomposition) with
            N = `strength` × **rmax** largest singular values removed for each
            angular order. This mimics the approach used in pBasex.

        In all cases, `strength` = 0 provides no regularization. For the
        Tikhonov methods, `strength` ~ 100 is a reasonable value for megapixel
        images. For truncated SVD, `strength` must be < 1; 0.1 is a reasonable
        value; `strength` ~ 0.5 can produce noticeable ringing artifacts.

    out : str or None
        shape of the output image:

        ``'same'`` (default):
            same shape and origin as the input
        ``'fold'`` (fastest):
            Q0 quadrant (for ``odd=False``) or right half (for ``odd=True``) up
            to **rmax**, but limited to the largest input-image quadrant (or
            half)
        ``'unfold'``:
            like ``'fold'``, but symmetrically “unfolded”
        ``'full'``:
            all pixels with radii up to **rmax**
        ``'full-unique'``:
            the unique part of ``'full'``: Q0 quadrant for ``odd=False``, right
            half for ``odd=True``
        ``None``:
            no image (**recon** will be ``None``). Can be useful to avoid
            unnecessary calculations if only the transformed radial
            distributions (**distr**) are needed.

    Returns
    -------
    recon : 2D numpy array or None
        the transformed image. Is centered and might have different dimensions
        than the input image.
    distr : Distributions.Results object
        the object from which various distributions for the transformed image
        can be retrieved, see :class:`abel.tools.vmi.Distributions.Results`
    """
    if order == 0:
        odd = False  # (to eliminate additional checks)
    elif order % 2:
        odd = True  # enable automatically for odd orders

    # extract radial profiles from input image
    p = _profiles(IM, origin, rmax, order, odd, weights)
    # (caches Distributions as _dst)

    Rmax = _dst.rmax

    # get appropriate transform matrices
    A = get_bs_cached(Rmax, order, odd, direction, reg)

    # transform radial profiles
    c = [An.dot(pn) for An, pn in zip(A, p)]

    # construct output (transformed) distributions
    distr = Distributions.Results(np.arange(Rmax + 1), np.array(c), order, odd)

    if out is None:
        return None, distr

    # output size
    if out == 'same':
        height = _dst.Qheight if odd else _dst.VER + 1
        width = _dst.HOR + 1
    elif out in ['fold', 'unfold']:
        height = _dst.Qheight
        width = _dst.Qwidth
    elif out in ['full', 'full-unique']:
        height = 2 * Rmax + 1 if odd else Rmax + 1
        width = Rmax + 1
    else:
        raise ValueError('Wrong output shape "{}"'.format(out))
    # construct output image from transformed radial profiles
    recon = _image(height, width, c)  # bottom right quadrant or right half
    if odd:
        if out not in ['fold', 'full-unique']:
            # combine with left half (mirrored without central column)
            recon = np.hstack((recon[:, :0:-1], recon))
    else:  # even only
        recon = recon[::-1]  # flip to Q0
        if out not in ['fold', 'full-unique']:
            # assemble full image
            recon = put_image_quadrants((recon, recon, recon, recon),
                                        (2 * height - 1, 2 * width - 1))
    if out == 'same':
        # crop as needed
        row = 0 if odd else _dst.VER - _dst.row
        col = _dst.HOR - _dst.col
        H, W = IM.shape
        recon = recon[row:row + H, col:col + W]

    return recon, distr


def _profiles(IM, origin, rmax, order, odd, weights):
    """
    Get radial profiles of cos^n theta terms from the input image.
    """
    # the Distributions object is cached to speed up further calculations,
    # plus its cos^n theta matrices are used later to construct the transformed
    # image
    global _prm, _dst, _ibs

    prm = [IM.shape, origin, rmax, order, odd, weights]
    if _prm != prm:
        _prm = prm
        _dst = Distributions(origin=origin, rmax=rmax, order=order, odd=odd,
                             weights=weights, use_sin=False, method='linear')
        _ibs = None

    return _dst(IM).cos()


def _get_image_bs(height, width):
    global _ibs

    if _ibs is not None:
        return _ibs

    # _dst quadrant has the minimal size, so height and width either equal its
    # dimensions, or at least one of them is larger
    if height == _dst.Qheight and width == _dst.Qwidth:
        # use arrays already computed in _dst
        _ibs = [_dst.bin, _dst.wl, _dst.wu, _dst.c]
    else:  # height > _dst.Qheight or width > _dst.Qwidth
        # compute arrays of requested size
        rmax = _dst.rmax
        # x row
        x = np.arange(float(width))
        # y and y^2 columns
        y0 = rmax if _dst.odd else 0
        y = y0 - np.arange(float(height))[:, None]
        y2 = y**2
        # arrays of r^2 and r
        r2 = x**2 + y2
        r = np.sqrt(r2)
        # radial bins
        rbin = r.astype(int)  # round down (floor)
        rbin[rbin > rmax] = rmax + 1  # last bin is then discarded
        # weights for upper and lower bins
        wu = r - rbin
        wl = 1 - wu
        # cos^n theta
        cos = [None]  # (cos^0 theta is not used)
        if _dst.odd:
            r[y0, 0] = np.inf  # (avoid division by zero)
            cos.append(y / r)  # cos^1 theta
        else:
            r2[0, 0] = np.inf  # (avoid division by zero)
            cos.append(y2 / r2)  # cos^2 theta
        for n in range(2, len(_dst.c)):  # remaining powers
            cos.append(cos[1] * cos[n - 1])

        _ibs = [rbin, wl, wu, cos]

    return _ibs


def _image(height, width, c):
    """
    Create transformed image (lower right quadrant for even-only,
    right half for odd) from its cos^n theta radial profiles.
    """
    rbin, wl, wu, cos = _get_image_bs(height, width)

    # 0th order (isotropic)
    IM = (wl * np.append(c[0], [0])[rbin] +  # lower bins
          wu * np.append(c[0][1:], [0, 0])[rbin])  # upper bins
    # add all other orders
    for cn, cosn in zip(c[1:], cos[1:]):
        IM += (wl * np.append(cn, [0])[rbin] +
               wu * np.append(cn[1:], [0, 0])[rbin]) * cosn
    # (weighting for each order is somehow faster than processing lower and
    #  upper bins separately and then combining)

    return IM


def _bs_rbasex(Rmax, order, odd):
    """
    Compute radial parts of basis projections for R and radii up to Rmax.
    """
    # all needed orders (even or all) from 0 to order
    orders = range(0, order + 1, 1 if odd else 2)

    # list of matrces for projections: P[i][R, r] = p_{R:n_i}(r),
    # sampled at r = 0, ..., R, for R = 0, ..., Rmax and orders n
    P = [np.eye(Rmax + 1, order='F') for n in orders]  # ('F' is faster here)
    # (p_{0;0}(0) = 1 indeed, but p_{0;n}(0) = 0 for all other orders,
    #  so P[i][0, 0] are also set to 1 to make P[i] nondegenerate)

    # fill p_{R>0;0}(0) = 2 (all other p_{R>0;n}(0) = 0 already)
    P[0][1:, 0] = 2

    # fill all other r > 0 columns (only functions with R >= r are non-zero)
    for r in range(1, Rmax + 1):
        # all needed R - 1, R, R + 1 for current r
        R = np.arange(r - 1, Rmax + 2, dtype=float)

        # rho = max(r, R)
        rho = R.copy()
        rho[0] = r

        # since z = sqrt(R^2 - r^2) for R >= r, otherwise 0,
        # it is z = sqrt(max(r, R)^2 - r^2) = sqrt(rho^2 - r^2)
        z = np.sqrt(rho**2 - r**2)

        f = r / rho  # "f" means "fraction r / rho"

        rln = r * np.log(z + rho)

        # define F_n for all needed n
        F = {-1: (z / f + rln) / 2, 0: z}
        if order >= 1:
            F[1] = rln
        if order >= 2:
            F[2] = r * np.arccos(f)
        if order >= 3:
            F[3] = z * f
        if order >= 4:
            fn = f.copy()  # current (r / rho)^n
            for n in range(2, order - 1):  # from 4 - 2 to order - 2
                fn *= f
                F[n + 2] = (z * fn + (n - 1) * F[n]) / n

        # compute p_{R;n}(r) for all needed R and n
        for i, n in enumerate(orders):
            rFRF = r * F[n - 1] - R * F[n]
            P[i][r:, r] = 2 * (2 * rFRF[1:-1] - rFRF[2:] - rFRF[:-2])
            #    R >= r                 at R         R - 1      R + 1

    return P


def get_bs_cached(Rmax, order=2, odd=False, direction='inverse', reg=None):
    """
    Internal function.

    Gets the basis set (from cache or runs computations)
    and calculates the transform matrix.
    Loaded/calculated matrices are also cached in memory.

    Parameters
    ----------
    Rmax : int
        largest radius to be transformed
    order : int
        highest angular order
    odd : bool
        include odd angular orders
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed
    reg : None or tuple (str, float)
        regularization type and strength for inverse transform
    Returns
    -------
    A : list of 2D numpy arrays
        (**Rmax** + 1) × (**Rmax** + 1) matrices of the Abel transform (forward
        or inverse) for each angular order
    """
    global _bs_prm, _bs, _trf, _tri_prm, _tri

    prm = [Rmax, order, odd]
    if _bs is None or _bs_prm != prm:
        _bs_prm = prm
        _bs = _bs_rbasex(Rmax, order, odd)
        _trf = None
        _tri_prm = None
        _tri = None

    if direction == 'forward':
        if _trf is None:
            _trf = _bs
        return _trf
    else:  # 'inverse'
        if _tri_prm != [reg]:
            _tri_prm = [reg]
            if reg is None:
                _tri_prm = [None]
                # P[n] are triangular, thus can be inverted faster than general
                # matrices, however, NumPy/SciPy do not have such functions;
                # nevertheless, solve_triangular() is ~twice faster than inv()
                # (and ...(Pn, I, lower=True).T is faster than ...(Pn.T, I))
                I = np.eye(Rmax + 1)
                _tri = [solve_triangular(Pn, I, lower=True).T for Pn in _bs]
            elif np.ndim(reg) == 0:  # not sequence type
                raise ValueError('Wrong regularization format "{}"'.
                                 format(reg))
            elif reg[0] == 'L2':  # Tikhonov L2 norm
                E = np.diag([reg[1]] * (Rmax + 1))
                # regularized inverse for each angular order
                _tri = [Pn.dot(inv((Pn.T).dot(Pn) + E)) for Pn in _bs]
            elif reg[0] == 'diff':  # Tikhonov derivative
                # GTG = reg D^T D, where D is 1st-order difference operator
                GTG = 2 * np.eye(Rmax + 1) - \
                          np.eye(Rmax + 1, k=-1) - \
                          np.eye(Rmax + 1, k=1)
                GTG[0, 0] = 1
                GTG[-1, -1] = 1
                GTG *= reg[1]
                # regularized inverse for each angular order
                _tri = [Pn.dot(inv((Pn.T).dot(Pn) + GTG)) for Pn in _bs]
            elif reg[0] == 'SVD':
                if reg[1] > 1:
                    raise ValueError('Wrong SVD truncation factor {} > 1'.
                                     format(reg[1]))
                # truncation index (smallest SV of P -> largest SV of inverse)
                rmax = int((1 - reg[1]) * Rmax) + 1
                _tri = []
                # loop over angular orders
                for Pn in _bs:
                    U, s, Vh = svd(Pn)
                    # truncate matrices
                    U = U[:, :rmax]
                    s = 1 / s[:rmax]  # inverse
                    Vh = Vh[:rmax]
                    # regularized inverse for this angular order
                    _tri.append((U * s).dot(Vh))
            else:
                raise ValueError('Wrong regularization type "{}"'.
                                 format(reg[0]))

        return _tri


def cache_cleanup(select='all'):
    """
    Utility function.

    Frees the memory caches created by ``get_bs_cached()``.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    select : str
        selects which caches to clean:

        ``all`` (default)
            everything, including basis;
        ``forward``
            forward transform;
        ``inverse``
            inverse transform.

    Returns
    -------
    None
    """
    global _prm, _dst, _bs_prm, _bs, _ibs, _trf, _tri_prm, _tri

    if select == 'all':
        _prm = None
        _dst = None
        _bs_prm = None
        _bs = None
        _ibs = None
    if select in ('all', 'forward'):
        _trf = None
    if select in ('all', 'inverse'):
        _tri_prm = None
        _tri = None
