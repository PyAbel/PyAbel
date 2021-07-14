# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys

import numpy as np
from scipy.linalg import inv, toeplitz
from scipy.optimize import nnls

from abel.tools.polynomial import PiecewisePolynomial as PP


def daun_transform(data, reg=0.0, order=0, verbose=True, direction='inverse'):
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
    verbose : bool
        determines whether progress report should be printed
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

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
        reg_type, strength = None, None
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
        recon = data.dot(M)

    if dim == 1:
        return recon[0]
    else:
        return recon


def get_bs_cached(n, order=0, reg_type='diff', strength=0, verbose=False,
                  direction='inverse'):
    """
    Internal function.

    Gets the basis set and calculates the necessary transform matrix.

    Parameters
    ----------
    n : int
        half-width of the image in pixels, must include the axial pixel
    order : int
        polynomial order for basis functions (0–3)
    reg_type: None or str
        regularization type (``None``, ``'diff'``, ``'L2'``, ``'nonneg'``)
    strength : float
        Tikhonov regularization parameter (for **reg_type** = ``'diff'`` and
        ``'L2'``, ignored otherwise)
    verbose : bool
        print some debug information
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    Returns
    -------
    M : n × n numpy array
        matrix of the Abel transform (forward or inverse)
    """
    A = _bs_daun(n, order, verbose)

    if direction == 'forward' or reg_type == 'nonneg':
        return A

    if reg_type is None or strength == 0:
        # simple inverse
        return inv(A)

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
    return A.T.dot(inv(A.dot(A.T) + strength * LTL))


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
    # pixel radii
    r = np.arange(float(n))

    # projections (Abel transforms) of given-order basis functions
    if order == 0:
        def p(j):
            return PP(r, [(j - 1/2, j + 1/2, [1], j)])
    elif order == 1:
        def p(j):
            return PP(r, [(j - 1, j, [1,  1], j),
                          (j, j + 1, [1, -1], j)])
    elif order == 2:
        def p(j):
            return PP(r, [(j - 1,   j - 1/2, [0, 0,  2], j - 1),
                          (j - 1/2, j + 1/2, [1, 0, -2], j),
                          (j + 1/2, j + 1,   [0, 0,  2], j + 1)])
    elif order == 3:
        def p(j):
            return PP(r, [(j - 1, j, [1, 0, -3, -2], j),
                          (j, j + 1, [1, 0, -3,  2], j)])

        # derivative basis functions
        def q(j):
            return PP(r, [(j - 1, j, [0, 1,  2, 1], j),
                          (j, j + 1, [0, 1, -2, 1], j)])
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
        A[j] = p(j).abel

    if order == 3:
        # coefficient matrix for derivative functions
        B = np.empty((n, n))
        for j in range(n):
            B[j] = q(j).abel
        # solve for smooth derivative and modify A accordingly
        C = toeplitz([4, 1] + [0] * (n - 2))
        C[1, 0] = C[-2, -1] = 0
        D = toeplitz([0, 3] + [0] * (n - 2), [0, -3] + [0] * (n - 2))
        D[1, 0] = D[-2, -1] = 0
        A += D.dot(inv(C)).dot(B)

    return A
