# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time
import os.path
from os import listdir
import re
from math import exp, log
import sys


import numpy as np
from scipy.special import gammaln
from scipy.linalg import inv

from abel.tools.polynomial import PiecewisePolynomial

from ._version import __version__

#############################################################################
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "Reconstruction of Abel-transformable images:
# The Gaussian basis-set expansion Abel transform method"
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler,
# Review of Scientific Instruments 73, 2634 (2002).
# https://doi.org/10.1063/1.1482156
#
#
# 2018-10-29
#   MR added forward transform.
# 2018-10-27
#   MR exposed BASIS_SET_CUTOFF for speed/accuracy control.
# 2018-10-07
#   MR added intensity correction.
#   Also smaller basis sets are now reused when generating larger ones.
#   Removed unnecessary scipy methods (scipy.dot is actually numpy.dot).
# 2018-10-03
#   MR completely rewrote basis generation (half-width, efficiency).
#   Switched from (n, nbf) to (n, sigma) basis specification;
#   nbf is now determined automatically, and nbf != n actually works.
#   The regularization parameter now differs from BASEX.exe by a factor
#   of 4/pi (BASEX.exe had an incorrect prefactor in basis projections).
# 2018-09-29
#   MR improved loading cached basis sets:
#   If the required basis is not available, but a larger compatible is,
#   then the latter will be loaded and cropped to the required size.
# 2018-09-19
#   MR switched to half-width transform.
#   The results now match BASEX.exe.
# 2018-09-08
#   MR enabled Tikhonov regularization.
#   Also, basis and transform matrices are now kept in memory
#   between invocations and reloaded/recalculated only when needed.
# Version 0.62 - 2016-03-07
#   DH changed all linear algebra steps from numpy to scipy
#   Scipy uses fortran fftpack, so it will be faster on some systems
#   or the same speed as numpy in most cases.
# Version 0.61 - 2016-02-17
#   Major reformatting - removed up/down symmetic version and
#   made basex work with only one quadrant.
# Version 0.6 - 2015-12-01
#   Another major code reformatting
# Version 0.5 - 2015-11-16
#   Code cleanup
# Version 0.4 - 2015-05-0
#   Major code update see pull request
#   https://github.com/DanHickstein/pyBASEX/pull/3
# Version 0.3 - 2015-02-01
#   Added documentation
# Version 0.2 - 2014-10-09
#   Adding a "center_and_transform" function to make things easier
# Versions 0.1 - 2012
#   First port to Python
#
#############################################################################


def basex_transform(data, sigma=1.0, reg=0.0, correction=True,
                    basis_dir='./', dr=1.0,
                    verbose=True, direction='inverse'):
    """
    This function performs the BASEX (BAsis Set EXpansion)
    Abel transform. It works on a "right side" image. I.e.,
    it works on just half of a cylindrically symmetric
    object, and ``data[0,0]`` should correspond to a central pixel.
    To perform a BASEX transform on a whole image, use ::

        abel.Transform(image, method='basex', direction='inverse').transform

    This BASEX implementation only works with images that have an
    odd-integer full width.

    Parameters
    ----------
    data : m × n numpy array
        the image to be transformed.
        ``data[:,0]`` should correspond to the central column of the image.
    sigma : float
        width parameter for basis functions, see equation (14) in the article.
        Determines the number of basis functions (**n**/**sigma** rounded).
        Can be any positive number, but using **sigma** < 1
        is not very meaningful and requires regularization.
    reg : float
        regularization parameter, square of the Tikhonov factor.

            ``reg=0`` means no regularization,

            ``reg=100`` is a reasonable value for megapixel images.

        Forward transform requires regularization only if **sigma** < 1,
        and **reg** should be ≪ 1.
    correction : boolean
        apply intensity correction in order to reduce method artifacts
        (intensity normalization and oscillations)
    basis_dir : str
        path to the directory for saving / loading the basis sets.
        If ``None``, the basis set will not be saved to disk.
    dr : float
        size of one pixel in the radial direction.
        This only affects the absolute scaling of the transformed image.
    verbose : boolean
        determines whether statements should be printed
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    Returns
    -------
    recon : m × n numpy array
        the transformed (half) image
    """

    # make sure that the data is the right shape (1D must be converted to 2D):
    data = np.atleast_2d(data)
    h, w = data.shape

    if data.shape[0] == 1:
        data_ndim = 1
    elif data.shape[1] == 1:
        raise ValueError('Wrong input shape for '
                         'data {0}, should be (N1, N2) '
                         'or (1, N), not (N, 1)'.format(data.shape))
    else:
        data_ndim = 2

    n = data.shape[1]

    # load the basis sets and compute the transform matrix
    A = get_bs_cached(n, sigma=sigma, reg=reg, correction=correction,
                      basis_dir=basis_dir, dr=dr, verbose=verbose,
                      direction=direction)

    # do the actual transform
    recon = basex_core_transform(data, A)

    if data_ndim == 1:  # taking the middle row, since the rest are zeroes
        recon = recon[recon.shape[0] - recon.shape[0]//2 - 1]  # ??
    if h == 1:
        return recon
    else:
        return recon


def basex_core_transform(rawdata, A):
    """
    Internal function that does the actual BASEX transform.
    It requires that the transform matrix be passed.

    Parameters
    ----------
    rawdata : m × n numpy array
        right half (with the axis) of the input image.
    A : n × n numpy array
        2D array given by the transform-calculation function

    Returns
    -------
    IM : m × n numpy array
        the Abel-transformed image
    """

    # Note that our images are stored with matrix rows corresponding to
    # horizontal image lines. This is consistent with the Matlab and C++
    # BASEX implementations, but is the opposite to the article notation.
    # Thus all matrices are transposed and multiplied backwards with
    # respect to the article.
    # The vertical transform is not applied, since without regularization
    # its overall effect is an identity transform.

    # transform the image
    return rawdata.dot(A)


def _get_A(M, Mc, reg, direction):
    """ Internal helper function.
        Calculates the forward/inverse Abel-transform matrix
        from basis matrices for given regularization parameter.
    """
    # A   is the overall (horizontal) transform matrix,
    #     corresponds to A^T Z in the article.
    # reg corresponds to q_1^2 in the article.

    # basis matrices for input and output spaces
    if direction == 'forward':
        Bi, Bo = Mc, M  # image -> projection
    else:  # 'inverse'
        Bi, Bo = M, Mc  # projection -> image

    n, nbf = M.shape
    if reg == 0.0 and nbf == n:
        A = inv(Bi.T).dot(Bo.T)
    else:
        # square of Tikhonov matrix
        E = np.diag([reg] * nbf)
        # regularized inverse of input basis
        R = Bi.dot(inv((Bi.T).dot(Bi) + E))
        # {expansion coefficients} = input . R
        # output = {expansion coefficients} . {output basis}
        # so: output = input . (R . {output basis})
        #     output = input . A
        #     A = R . {output basis}
        # A is the matrix of the Abel transform
        A = R.dot(Bo.T)

    return A


def _nbf(n, sigma):
    """
    Internal helper function.
    Calculates the number of basis functions **nbf** from
    the half-image width **n** and the basis width parameter **sigma**.
    """
    return int(round(n / sigma))


# Cached matrices and their parameters
# basis set
_bs_prm = None   # [n, sigma]
_bs = None       # [M, Mc]
# forward transform
_trf_prm = None  # [reg, correction, dr]
_trf = None      # Af
# inverse transform
_tri_prm = None  # [reg, correction, dr]
_tri = None      # Ai

def get_bs_cached(n, sigma=1.0, reg=0.0, correction=True,
                  basis_dir='.', dr=1.0, verbose=False, direction='inverse'):
    """
    Internal function.

    Gets BASEX basis sets, using the disk as a cache
    (i.e. load from disk if they exist,
    if not, calculate them and save a copy on disk)
    and calculates the transform matrix.
    To prevent saving the basis sets to disk, set ``basis_dir=None``.
    Loaded/calculated matrices are also cached in memory.

    Parameters
    ----------
    n : int
        Abel transform will be performed on an **n** pixels wide area
        of the (half) image
    sigma : float
        width parameter for basis functions
    reg : float
        regularization parameter
    correction : boolean
        apply intensity correction.
        Corrects wrong intensity normalization (seen for narrow basis sets),
        intensity oscillations (seen for broad basis sets),
        and intensity drop-off near *r* = 0 due to regularization.
    basis_dir : str
        path to the directory for saving / loading the basis sets.
        If ``None``, the basis sets will not be saved to disk.
    dr : float
        pixel size. This only affects the absolute scaling of the output.
    verbose : boolean
        determines whether statements should be printed
    direction : str: ``'forward'`` or ``'inverse'``
        type of Abel transform to be performed

    Returns
    -------
    A : n × n numpy array
        matrix of the Abel transform (forward or inverse)
    """

    global _bs_prm, _bs, _trf_prm, _trf, _tri_prm, _tri

    sigma = float(sigma)  # (ensure FP format)
    nbf = _nbf(n, sigma)
    M = None
    # Check whether basis for these parameters is already loaded
    if _bs_prm == [n, sigma]:
        if verbose:
            print('Using memory-cached basis sets')
        M, Mc = _bs
    else:  # try to load basis
        if basis_dir is not None:
            basis_file = 'basex_basis_{}_{}.npy'.format(n, sigma)

            def full_path(file_name):
                return os.path.join(basis_dir, file_name)

            # Try to find a suitable existing basis set
            if os.path.exists(full_path(basis_file)):
                # have exactly needed
                best_file = basis_file
            else:
                # Find the best (smallest among sufficient)
                # and the largest (to extend if not sufficient)
                best_file = None
                best_n = sys.maxsize
                largest_file = None
                largest_n = 0
                mask = re.compile(r'basex_basis_(\d+)_{}\.npy$'.format(sigma))
                for f in listdir(basis_dir):
                    # filter BASEX basis files
                    match = mask.match(f)
                    if not match:
                        continue
                    # extract basis image size (sigma was fixed above)
                    f_n = int(match.group(1))
                    # must be large enough and smaller than previous best
                    if f_n >= n and f_n < best_n:
                        # remember as new best
                        best_file = f
                        best_n = f_n
                    # largest must be just larger than previous
                    if f_n > largest_n:
                        # remember as new largest
                        largest_file = f
                        largest_n = f_n

            # If found, try to use it
            if best_file:
                if verbose:
                    print('Loading basis sets...')
                    # saved as a .npy file
                try:
                    M, Mc, M_version = np.load(full_path(best_file))
                    # crop if loaded larger
                    if M.shape != (n, nbf):
                        M = M[:n, :nbf]
                        Mc = Mc[:n, :nbf]
                        if verbose:
                            print('(cropped from {})'.format(best_file))
                except ValueError:
                    print('Cached basis file incompatible.')

        if M is None:  # generate the basis set
            if verbose:
                print('A suitable basis set was not found.',
                      'A new basis set will be generated.',
                      'This may take a few minutes.', sep='\n')
                if basis_dir is not None:
                    print('But don\'t worry, '
                          'it will be saved to disk for future use.')

            # Try to extend the largest available
            try:
                oldM, oldMc, M_version = np.load(full_path(largest_file))
                if verbose:
                    print('(extending {})'.format(largest_file))
            except:
                oldM = None  # (old Mc is not needed)

            M, Mc = _bs_basex(n, sigma, oldM, verbose=verbose)

            if basis_dir is not None:
                np.save(full_path(basis_file),
                        (M, Mc, np.array(__version__)))
                if verbose:
                    print('Basis set saved for later use to')
                    print('  {}'.format(basis_file))

        _bs_prm = [n, sigma]
        _bs = [M, Mc]
        _trf_prm = None
        _tri_prm = None

    # Check whether transform matrices for these parameters
    # are already created
    if direction == 'forward' and _trf_prm == [reg, correction, dr]:
        A = _trf
    elif direction == 'inverse' and _tri_prm == [reg, correction, dr]:
        A = _tri
    else:  # recalculate
        if verbose:
            print('Updating regularization...')
        A = _get_A(*_bs, reg=reg, direction=direction)
        if correction:
            if verbose:
                print('Calculating correction...')
            cor = get_basex_correction(A, sigma, direction)
            A = np.multiply(A, cor)
        if direction == 'forward':
            _trf_prm = [reg, correction, dr]
            _trf = A
        else:  # 'inverse'
            _tri_prm = [reg, correction, dr]
            _tri = A
        # apply intensity scaling, if needed
        if dr != 1.0:
            if direction == 'forward':
                A *= dr
            else:  # 'inverse'
                A /= dr

    return A


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
    global _bs_prm, _bs, _trf_prm, _trf, _tri_prm, _tri

    if select == 'all':
        _bs_prm = None
        _bs = None
    if select in ('all', 'forward'):
        _trf_prm = None
        _trf = None
    if select in ('all', 'inverse'):
        _tri_prm = None
        _tri = None


def get_basex_correction(A, sigma, direction):
    """
    Internal function.

    The default BASEX basis and the way its projection is calculated
    leads to artifacts in the reconstructed distribution --
    incorrect overall intensity for **sigma** = 1,
    intensity oscillations for other **sigma** values,
    intensity fluctuations (and drop-off for **reg** > 0) near *r* = 0.
    This function generates the intensity correction profile
    from the BASEX result for a step function with a soft edge (to avoid
    ringing) aligned with the last basis function.

    Parameters
    ----------
    A : n × n numpy array
        matrix of the Abel transform
    sigma : float
        basis width parameter
    direction : str: ``'forward'`` or ``'inverse'``
        type of the Abel transform

    Returns
    -------
    cor : 1 × n numpy array
        intensity correction profile
    """
    n = A.shape[0]
    nbf = _nbf(n, sigma)

    # Generate soft step function and its projection
    r = np.arange(float(n))
    # edge center, aligned with the last basis function
    c = (nbf - 0.5) * sigma
    # soft-edge halfwidth
    w = sigma
    # soft step: stitched constant (shelf) and 2 parabolas (soft edge)
    step = PiecewisePolynomial(r, [(0, c - w, [1]),
                                   (c - w, c, [1, 0, -1/2], c - w, w),
                                   (c, c + w, [0, 0, 1/2], c + w, w)])
    # (this is more numerically stable at large r than cubic smoothstep)

    # get BASEX Abel transform of the step
    # and set correction profile = expected / BASEX result
    if direction == 'forward':
        tran = basex_core_transform(step.func, A)
        cor = step.abel / tran
    else:  # 'inverse'
        tran = basex_core_transform(step.abel, A)
        cor = step.func / tran

    return cor


# The analytical expresion for the k-th basis-function projection
# involves a sum of k^2 terms, most of which are very small.
# Setting BASIS_SET_CUTOFF = c truncates this sum to ±cu terms
# around the maximum (u = x / sigma).
# The computation time is roughly proportional to this parameter,
# while the accuracy (at least for n, k < 10000) is as follows:
#   cutoff   relative error
#     4        < 2e-4
#     5        < 2e-6
#     6        < 7e-9
#     7        < 1e-11
#     8        < 6e-15
#     9        < 1e-15
# The last one reaches the 64-bit floating-point precision,
# so going beyond that is useless.
# See https://github.com/PyAbel/PyAbel/issues/230
BASIS_SET_CUTOFF = 9  # numerically exact

def _bs_basex(n=251, sigma=1.0, oldM=None, verbose=True):
    """
    Generates horizontal basis sets for the BASEX method.

    Parameters
    ----------
    n : int
        horizontal dimensions of the half-width image in pixels.
        Must include the axial pixel.
        See https://github.com/PyAbel/PyAbel/issues/34
    sigma : float
        width parameter for basis functions
    oldM : numpy array
        projected basis matrix for the same **sigma** but a smaller image size.
        Can be supplied to avoid recalculating matrix elements
        that are already available.

    Returns
    -------
    M, Mc : n × nbf numpy array
        Mc
            is the reconstructed-image basis rho_k(r_i) (~Gaussians),
            corresponds to Z^T in the article.
        M
            is the projected basis chi_k(x_i),
            corresponds to X^T in the article.
    """

    sigma = float(sigma)  # (ensure FP type)
    nbf = _nbf(n, sigma)  # number of basis functions

    if verbose:
        print('Generating horizontal BASEX basis sets for '
              'n = {}, sigma = {} (nbf = {}):'.format(n, sigma, nbf))
        print('k = 0...', end='')
        sys.stdout.flush()

    # Precompute tables of ln Gamma(...) terms;
    # notice that index i corresponds to argument i + 1 (and i + 1/2).
    maxk2 = (nbf - 1)**2
    # for Gamma(k^2 + 1) and Gamma(l + 1)
    lngamma = gammaln(np.arange(maxk2 + 1) + 1)
    # for Gamma(k^2 - l + 1) - Gamma(k^2 - l + 1/2)
    Dlngamma = lngamma - gammaln(np.arange(maxk2 + 1) + 1/2)

    # reduced coordinates u = x/sigma (or r/sigma) and their squares
    U = np.arange(float(n)) / sigma
    U2 = U * U

    Mc = np.empty((n, nbf))
    M = np.empty((n, nbf))
    # (indexing is Mc[r, k], M[x, k])
    old_n, old_nbf = 0, 0
    # reuse old elements, if available
    if oldM is not None:
        old_n, old_nbf = oldM.shape
        M[:old_n, :old_nbf] = oldM / sigma  # (full M will be *= sigma later)

    # Cases k = 0 and x = 0 (r = 0) are special, since general expressions
    # are valid only if considered as limits; here they are computed
    # separately, using expressions that result from taking these limits.
    # In all cases the sigma factor in projections is applied afterwards.

    # rho_0(r) = exp(-u^2)
    Mc[:, 0] = np.exp(-U2)
    # chi_0(x) = sqrt(pi) sigma exp(-u^2) = Gamma(1/2) sigma exp(-u^2)
    M[:, 0] = np.exp(gammaln(1/2) - U2)

    for k in range(1, nbf):
        k2 = k * k
        # prefactor ln[(e/k^2)^(k^2)]
        ek = (1 - log(k2)) * k2

        # Basis function rho_k(r)
        Mc[0, k] = 0
        Mc[1:, k] = np.exp(ek + np.log(U[1:]) * 2 * k2 - U2[1:])

        # Projected basis function chi_k(x)
        # full range of l
        L = np.arange(k2 + 1)
        # all ln Gamma(...) terms
        G = lngamma[k2] - lngamma[L] - Dlngamma[k2 - L]
        # Calculate chi_k(x) at each x_i
        M[0, k] = exp(ek + G[0])  # (u^(2l) = 1 for l = 0, otherwise 0)
        for i, u2 in enumerate(U2[1:], 1):
            # skip what was already filled
            if i < old_n and k < old_nbf:
                continue
            u = U[i]
            if u > k + 8:  # beyond outer shoulder
                M[i, k] = 0.0
                continue
            # index of the largest component
            lmax = min(int(u2), k2)
            # halfwidth of the important range
            delta = int(BASIS_SET_CUTOFF * (u + 2))
            # summation limits: ±delta from the maximum, but within [0, k^2]
            minl = max(0, lmax - delta)
            maxl = min(lmax + delta, k2)
            # list of ln[u^(2l)]
            lnu2L = log(u2) * L[minl:maxl+1]
            # sum over the important range
            M[i, k] = np.exp(ek - u2 + G[minl:maxl+1] + lnu2L).sum()

        if verbose and k % 50 == 0:
            print('{}...'.format(k), end='')
            sys.stdout.flush()

    if verbose:
        print(k + 1)

    M *= sigma  # applying the sigma factor

    return M, Mc
