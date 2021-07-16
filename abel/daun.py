# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
from scipy.linalg import inv, toeplitz, solve_banded, solve_triangular
from scipy.optimize import nnls


def daun_transform(data, reg=0.0, order=0, direction='inverse', verbose=True):
    """
    Forward and inverse Abel transforms based on onion-peeling deconvolution
    using Tikhonov regularization described in

    K. J. Daun, K. A. Thomson, F. Liu, G. J. Smallwood,
    "Deconvolution of axisymmetric flame properties using Tikhonov
    regularization",
    `Appl. Opt. 45, 4638–4646 (2006)
    <https://doi.org/10.1364/AO.45.004638>`__.

    with additional basis-function types and regularization methods.

    This function operates on the “right side” of an image, that it, just one
    half of a cylindrically symmetric image, with the axial pixels located in
    the 0-th column.

    Parameters
    ----------
    data : m × n numpy array
        the image to be transformed.
        ``data[:, 0]`` should correspond to the central column of the image.
    reg : float or tuple or str
        regularization for the inverse transform:

        ``strength``:
            same as ``('diff', strength)``
        ``('diff', strength)``:
            Tikhonov regularization using the first-order difference operator
            (first-derivative approximation), as described in the original
            article
        ``('L2', strength)``:
            Tikhonov regularization using the :math:`L_2` norm, like in
            :ref:`BASEX`
        ``('L2c', strength)``:
            same as ``('L2', strength)``, but with an intensity correction
            applied to compensate the drop near the symmetry axis
        ``'nonneg'``:
            non-negative least-squares solution.

            `Warning: this regularization method is very slow, typically taking
            up to a minute for a megapixel image.`
    order : int
        order of basis-function polynomials:

        0:
            rectangular functions (step-function approximation), corresponding
            to “onion peeling” from the original article
        1:
            triangular functions (piecewise linear approximation)
        2:
            piecewise quadratic functions (smooth approximation)
        3:
            piecewise cubic functions (cubic-spline approximation)
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed
    verbose : bool
        determines whether progress report should be printed

    Returns
    -------
    recon : m × n numpy array
        the transformed (half) image
    """
    # make sure that the data has the right shape (1D must be converted to 2D)
    # and type:
    dim = len(data.shape)
    data = np.atleast_2d(data).astype(float)
    h, w = data.shape

    if reg in [None, 0]:
        reg_type, strength = None, 0
    elif reg == 'nonneg':
        reg_type, strength = reg, None
    elif isinstance(reg, (float, int)):
        reg_type, strength = 'diff', reg
    elif np.ndim(reg) == 0 and reg != 'nonneg':
        raise ValueError('Wrong regularization format "{}"'.format(reg))
    else:
        reg_type, strength = reg

    # load the basis sets and compute the transform matrix
    M = get_bs_cached(w, order=order, reg_type=reg_type, strength=strength,
                      verbose=verbose, direction=direction)

    if reg == 'nonneg':
        if verbose:
            print('Solving NNLS equations...')
            sys.stdout.flush()
        recon = np.empty_like(data)
        # transform row by row
        for i in range(h):
            if verbose:
                print('\r{}/{}'.format(1 + i, h), end='')
                sys.stdout.flush()
            recon[i] = nnls(M.T, data[i])[0]
        if verbose:
            print('\nDone!')
    else:
        # do the linear transform
        if direction == 'inverse' and strength == 0 and order != 3:
            # (this is faster than general-purpose multiplication by inverse)
            recon = solve_triangular(M.T, data.T).T  # (here M is forward)
        else:
            recon = data.dot(M)

    if dim == 1:
        return recon[0]
    else:
        return recon


# Caches and their parameters
_bs = None  # basis set
_bs_prm = None  # [size, order]
_tr = None  # inverse-transform matrix
_tr_prm = None  # [size, type, strength]


def get_bs_cached(n, order=0, reg_type='diff', strength=0, direction='inverse',
                  verbose=False):
    """
    Internal function.

    Gets the basis set and calculates the necessary transform matrix (notice
    that inverse direction with ``'nonneg'`` regularization, as well as with
    **strength** = 0 for **order** ≠ 3, gives the forward (triangular) matrix,
    to be used in solvers).

    Parameters
    ----------
    n : int
        half-width of the image in pixels, must include the axial pixel
    order : int
        polynomial order for basis functions (0–3)
    reg_type: None or str
        regularization type (``None``, ``'diff'``, ``'L2'``, ``'L2c'``,
        ``'nonneg'``)
    strength : float
        Tikhonov regularization parameter (for **reg_type** = ``'diff'`` and
        ``'L2'``/``'L2c'``, ignored otherwise)
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed
    verbose : bool
        print some debug information

    Returns
    -------
    M : n × n numpy array
        matrix of the Abel transform (forward or inverse)
    """
    global _bs, _bs_prm, _tr, _tr_prm

    bs_OK = (_bs is not None and  # cached
             _bs_prm[1] == order and  # correct order
             (_bs_prm[0] == n if order == 3 else _bs_prm[0] >= n))  # good size
    if not bs_OK:
        # generate and cache
        _bs = _bs_daun(n, order, verbose)
        _bs_prm = [n, order]
        # reset cached inverse-transform matrix
        _tr = None
        _tr_prm = None

    if direction == 'forward' or reg_type == 'nonneg':
        return _bs[:n, :n]  # (cropping if too large)

    if reg_type is None:
        strength = 0

    if strength == 0:
        if _tr is None or _tr_prm[2] != 0:  # not cached or for strength != 0
            # compute transform matrix (full-sized, just in case)
            if order == 3:
                # general-purpose inverse of non-triangular
                _tr = inv(_bs)
            else:
                # triangular matrix as is — will be used in solve_triangular,
                # which is even faster than multiplication by cached inverse
                _tr = _bs
            _tr_prm = [_bs_prm[0], reg_type, strength]
        return _tr[:n, :n]  # (cropping if too large — safe for triangular)

    if _tr_prm != [n, reg_type, strength]:
        # square of Tikhonov matrix
        if reg_type == 'diff':
            # of difference operator (approx. derivative operator)
            LTL = toeplitz([2, -1] + [0] * (n - 2))
            LTL[0, 0] = LTL[-1, -1] = 1
        else:
            # for L2 norm
            LTL = np.eye(n)

        # regularized inverse
        # (transposed compared to the Daun article, since our data are in rows)
        A = _bs[:n, :n]
        _tr = A.T.dot(inv(A.dot(A.T) + strength * LTL))
        if reg_type == 'L2c':
            # apply correction: divide by regularized inverse of
            # forward-transformed uniform distribution
            _tr /= A.sum(axis=0).dot(_tr)
        _tr_prm = [n, reg_type, strength]
    return _tr


def cache_cleanup(select='all'):
    """
    Utility function.

    Frees the memory caches created by :func:`get_bs_cached`.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    select : str
        selects which caches to clean:

        ``'all'`` (default)
            everything, including basis set
        ``'inverse'``
            only inverse transform

    Returns
    -------
    None
    """
    global _bs, _bs_prm, _tr, _tr_prm

    if select == 'all':
        _bs = None
        _bs_prm = None
    _tr = None
    _tr_prm = None


def _bs_daun(n, order=0, verbose=False):
    """
    Internal function.

    Generate the projected basis set.

    Parameters
    ----------
    n : int
        half-width of the image in pixels, must include the axial pixel
    order : int
        polynomial order for basis functions (0–3)

    Returns
    -------
    A : n × n numpy array
        coefficient matrix (transposed projected basis set)
    """
    # pixel coordinates
    x = np.arange(float(n))
    # (and common subexpressions for projections)
    x2 = x**2
    if order > 0:
        x2logx = x2 * np.log(x, np.zeros_like(x), where=x > 0)

    # projections (Abel transforms) of given-order basis functions
    if order == 0:
        def p(j):
            o = np.zeros(n)
            o[:j + 1] += np.sqrt((j + 1/2)**2 - x2[:j + 1])
            if j > 0:
                o[:j] -= np.sqrt((j - 1/2)**2 - x2[:j])
            return 2 * o
    elif order == 1:
        def p(j):
            def P(R):
                y = np.sqrt(R**2 - x2[:R])
                return y * R - x2[:R] * np.log(y + R)
            o = np.zeros(n)
            o[:j + 1] += P(j + 1)
            o[:j] -= 2 * P(j)
            o[j] += x2logx[j]
            if j > 0:
                o[:j - 1] += P(j - 1)
                o[j - 1] -= x2logx[j - 1]
            return o
    elif order == 2:
        def p(j):
            def P(R, a, b, c):
                x2_R = x2[:int(R + 1/2)]
                y = np.sqrt(R**2 - x2_R)
                return y * (a * 2 + b * R + c * 4/3 * (R**2 / 2 + x2_R)) + \
                       b * x2_R * np.log(y + R)
            o = np.zeros(n)
            o[:j + 1] += P(j + 1, 2 * (j + 1)**2, -4 * (j + 1), 2) - \
                         P(j + 1/2, (2 * j + 1)**2, -4 * (2 * j + 1), 4)
            if j > 0:
                o[:j] += P(j - 1/2, (2 * j - 1)**2, -4 * (2 * j - 1), 4)
                o[j] -= 4 * j * x2logx[j]
                o[:j - 1] -= P(j - 1, 2 * (j - 1)**2, -4 * (j - 1), 2)
                o[j - 1] += 4 * (j - 1) * x2logx[j - 1]
            return o
    elif order == 3:
        def p(j):
            def P(R, a, b, c, d):
                y = np.sqrt(R**2 - x2[:R])
                return y * (a * 2 + (b + (c * 2/3 + d * R / 2) * R) * R +
                            (d * 3/4 * R + c * 4/3) * x2[:R]) + \
                       (b + d * 3/4 * x2[:R]) * x2[:R] * np.log(y + R)
            o = np.zeros(n)
            o[:j + 1] += P(j + 1,
                           -j**2 * (2 * j + 3) + 1, 6 * j * (j + 1),
                           -3 * (2 * j + 1),  2)
            o[:j] += P(j, 4 * j**3, -12 * j**2, 12 * j, -4)
            o[j] -= (6 * j * (j + 1) + 3/2 * x2[j]) * x2logx[j]
            if j > 0:
                o[:j - 1] -= P(j - 1,
                               j**2 * (2 * j - 3) + 1, -6 * j * (j - 1),
                               3 * (2 * j - 1), -2)
                o[j - 1] += 6 * (j * (j - 1) + x2[j - 1] / 4) * x2logx[j - 1]
            return o

        # derivative basis functions
        def q(j):
            def P(R, a, b, c, d):
                y = np.sqrt(R**2 - x2[:R])
                return y * (a * 2 + (b + (c * 2/3 + d * R / 2) * R) * R +
                            (d * 3/4 * R + c * 4/3) * x2[:R]) + \
                       (b + d * 3/4 * x2[:R]) * x2[:R] * np.log(y + R)
            o = np.zeros(n)
            o[:j + 1] += P(j + 1,
                           -j * (j * (j + 2) + 1), j * (3 * j + 4) + 1,
                           -3 * j - 2, 1)
            o[:j] += P(j, 4 * j**2, -8 * j, 4, 0)
            o[j] -= (j * (3 * j + 4) + 1 + 3/4 * x2[j]) * x2logx[j]
            if j > 0:
                o[:j - 1] -= P(j - 1,
                               -j * (j * (j - 2) + 1), j * (3 * j - 4) + 1,
                               -3 * j + 2, 1)
                o[j - 1] -= (j * (3 * j - 4) + 1 + 3/4 * x2[j - 1]) * \
                            x2logx[j - 1]
            return o
    else:
        raise ValueError('Wrong order={} (must be 0, 1, 2 or 3).'.
                         format(repr(order)))

    if verbose:
        print('Generating basis projections for '
              'n = {}, order = {}...'.format(n, order))

    # fill the coefficient matrix
    # (transposed compared to the Daun article, since our data are in rows)
    A = np.empty((n, n))
    for j in range(n):
        A[j] = p(j)

    if order == 3:
        # coefficient matrix for derivative functions
        B = np.empty((n, n))
        for j in range(n):
            B[j] = q(j)
        # solve for smooth derivative and modify A accordingly
        C = solve_banded((1, 1), ([0] + [1] * (n - 2) + [0],
                                  [4] * n,
                                  [0] + [1] * (n - 2) + [0]),
                         3 * B)[1:-1, 1:-1]
        A[2:,  1:-1] += C
        A[:-2, 1:-1] -= C

    return A
