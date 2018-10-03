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
from scipy.ndimage import median_filter, gaussian_filter, center_of_mass
import scipy

from ._version import __version__

#############################################################################
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "The Gaussian basis-set expansion Abel transform method"
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler,
# Review of Scientific Instruments 73, 2634 (2002).
#
#
# 2018-10-03
#   MR completely rewrote basis generation (half-width, efficiency).
#   Switched from (n, nbf) to (n, sigma) basis specification;
#   nbf is now determined automatically, and nbf != n actually works.
# 2018-09-29
#   MR improved loading cached basis sets:
#   If the required basis is not available, but a larger compatible is,
#   then the latter will be loaded and cropped to the required size.
# 2018-09-25
#   MR added basis correction near r = 0
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


def basex_transform(data, sigma=1.0, reg=0.0, bs_correction=False,
                    basis_dir='./', dr=1.0,
                    verbose=True, direction='inverse'):
    """
    This function performs the BASEX (BAsis Set EXpansion)
    Abel transform. It works on a "right side" image. I.e.,
    it works on just half of a cylindrically symmetric
    object, and ``data[0,0]`` should correspond to a central pixel.
    To perform a BASEX transorm on
    a whole image, use ::

        abel.Transform(image, method='basex', direction='inverse').transform

    This BASEX implementation only works with images that have an
    odd-integer full width.

    Note: only the `direction="inverse"` transform is currently implemented.


    Parameters
    ----------
    data : an m x n numpy array
        the image to be transformed.
        ``data[:,0]`` should correspond to the central column of the image.
    sigma : float
        width parameter for basis functions, see equation (14) in the article.
        Determines the number of basis functions (``n / sigma`` rounded).
        Can be any positive number, but using sigma < 1 is not very meaningful
        and requires regularization.
    reg : float
        regularization parameter, square of the Tikhonov factor.
        ``reg=0`` means no regularization,
        ``reg=100`` is a reasonable value for megapixel images.
    bs_correction : boolean
        apply a correction to k = 0 basis functions in order to reduce
        the artifact near r = 0.
    basis_dir : str
        path to the directory for saving / loading the basis set coefficients.
        If None, the basis set will not be saved to disk.
    dr : float
        size of one pixel in the radial direction.
        This only affects the absolute scaling of the transformed image.
    verbose : boolean
        determines whether statements should be printed.
    direction : str
        the type of Abel transform to be performed.
        Currently only accepts value ``'inverse'``.


    Returns
    -------
    recon : m x n numpy array
        the transformed (half) image

    """

    if direction != 'inverse':
        raise ValueError('Forward BASEX transform not implemented')

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

    # load the basis sets:
    Ai = get_bs_basex_cached(n, sigma=sigma, reg=reg,
                             bs_correction=bs_correction,
                             basis_dir=basis_dir, verbose=verbose)

    # Do the actual transform:
    recon = basex_core_transform(data, Ai, dr)

    if data_ndim == 1:  # taking the middle row, since the rest are zeroes
        recon = recon[recon.shape[0] - recon.shape[0]//2 - 1]  # ??
    if h == 1:
        return recon
    else:
        return recon


def basex_core_transform(rawdata, Ai, dr=1.0):
    """
    This is the internal function that does the actual BASEX transform.
    It requires that the transform matrix be passed.


    Parameters
    ----------
    rawdata : m x n numpy array
        the raw image. This is the right half (with the axis) of the image.
    Ai : n x n numpy array
        2D array given by the transform-calculation function
    dr : float
        pixel size. This only affects the absolute scaling of the output.


    Returns
    -------
    IM : m x n numpy array
        The Abel-transformed image, a slice of the 3D distribution
    """

    # Note that our images are stored with matrix rows corresponding to
    # horizontal image lines. This is consistent with the Matlab and C++
    # BASEX implementations, but is the opposite to the article notation.
    # Thus all matrices are transposed and multiplied backwards with
    # respect to the article.
    # The vertical transform is not applied, since without regularization
    # its overall effect is an identity transform.

    # Reconstructing image  - This is where the magic happens
    IM = scipy.dot(rawdata, Ai) / dr
    # P = dot(dot(Mc,Ci),M.T) # This calculates the projection, !! not
    # which should recreate the original image                  !! really
    return IM


def _get_Ai(M, Mc, reg):
    """ An internal helper function for no-up/down-asymmetry BASEX:
        given basis sets M, Mc,
        return matrix of inverse Abel transform
    """
    # Ai  is the overall (horizontal) transform matrix,
    #     corresponds to A^T Z in the article.
    # reg corresponds to q_1^2 in the article.

    n, nbf = M.shape
    if reg == 0.0 and nbf == n:
        Ai = inv(M.T).dot(Mc.T)
    else:
        # square of Tikhonov matrix
        E = np.diag([reg] * nbf)
        # regularized inverse of basis projection
        R = M.dot(inv((M.T).dot(M) + E))
        # {expansion coefficients} = projection . R
        # image = {expansion coefficients} . {image basis}
        # so: image = projection . (R . {image basis})
        #     image = projection . Ai
        #     Ai = R . {image basis}
        # Ai is the matrix of the inverse Abel transform
        Ai = R.dot(Mc.T)

    return Ai


def _nbf(n, sigma):
    """
    Internal helper function.
    Calculates the number of basis functions ``nbf`` from
    the half-image width ``n`` and the basis width parameter ``sigma``.
    """
    return int(round(n / sigma))


# Cached matrices and their parameters
_prm = None  # [n, sigma, bs_correction]
_M = None    # [M, Mc]
_reg = None  # reg
_Ai = None   # Ai

def get_bs_basex_cached(n, sigma=1.0, reg=0.0, bs_correction=False,
                        basis_dir='.', verbose=False):
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
        Abel inverse transform will be performed on an
        ``n`` pixels wide area of the (half) image
    sigma : float
        width parameter for basis functions
    reg : float
        regularization parameter
    bs_correction : boolean
        apply a correction to k = 0 functions
    basis_dir : str
        path to the directory for saving / loading the basis sets.
        If None, the basis sets will not be saved to disk.

    Returns
    -------
    Ai: numpy array
        the matrix of the inverse Abel transform.
    """

    global _prm, _M, _reg, _Ai

    sigma = float(sigma)  # (ensure FP format)
    nbf = _nbf(n, sigma)
    M = None
    # Check whether basis for these parameters is already loaded
    if _prm == [n, sigma, bs_correction]:
        if verbose:
            print('Using memory-cached basis sets')
        M, Mc = _M
    else:  # try to load basis
        if basis_dir is not None:
            basis_name = 'basex_basis_{}_{}.npy'.format(n, sigma)
            path_to_basis_file = os.path.join(basis_dir, basis_name)

            # Try to find a suitable existing basis set
            if os.path.exists(path_to_basis_file):  # have exactly needed
                best_file = path_to_basis_file
            else:
                # Find the best (smallest among sufficient)
                best_file = None
                best_n = sys.maxint
                for f in listdir(basis_dir):
                    # filter BASEX basis files
                    match = re.match(r'basex_basis_(\d+)_(\d+\.\d+).npy$', f)
                    if not match:
                        continue
                    # extract basis parameters
                    f_n, f_sigma = int(match.group(1)), float(match.group(2))
                    # must be large enough and smaller than previous best
                    if f_n < n or f_n > best_n:
                        continue
                    # must have the same sigma
                    if f_sigma == sigma:
                        # remember as new best
                        best_file = f
                        best_n = f_n

            # If found, try to use it
            if best_file:
                if verbose:
                    print('Loading basis sets...')
                    # saved as a .npy file
                try:
                    M, Mc, M_version = np.load(best_file)
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

            M, Mc = _bs_basex(n, sigma, verbose=verbose)

            if basis_dir is not None:
                np.save(path_to_basis_file,
                        (M, Mc, np.array(__version__)))
                if verbose:
                    print('Basis set saved for later use to')
                    print('  {}'.format(path_to_basis_file))

        # Apply basis correction
        if bs_correction:
            # This is a dirty hack!  ?? what if sigma != 1.0?
            # See https://github.com/PyAbel/PyAbel/issues/230
            l = min(nbf, 5)  # modifying at most 5 first points (what fits)
            # image basis function k = 0
            Mc[:l, 0] = [1.27, 0.19, -0.025, -0.015, -0.007][:l]
            # its projection
            M[:l, 0] = [1.65, 0.18, -0.15, -0.09, -0.04][:l]

        _prm = [n, sigma, bs_correction]
        _M = [M, Mc]
        _reg = None

    # Check whether transform matrices for this regularization
    # are already loaded
    if _reg == reg:
        Ai = _Ai
    else:  # recalculate
        if verbose:
            print('Updating regularization...')
        Ai = _get_Ai(*_M, reg=reg)
        _reg = reg
        _Ai = Ai

    return Ai


def basex_cleanup():
    """
    Utility function.

    Frees the memory caches created by ``get_bs_basex_cached()``.
    This is usually pointless, but might be required after working
    with very large images, if more RAM is needed for further tasks.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    global _prm, _M, _reg, _Ai

    _prm = None
    _M = None
    _reg = None
    _Ai = None


def _bs_basex(n=251, sigma=1.0, verbose=True):
    """
    Generates horizontal basis sets for the BASEX method.

    Parameters:
    -----------
    n : integer
        horizontal dimensions of the half-width image in pixels.
        Must include the axial pixel.
        See https://github.com/PyAbel/PyAbel/issues/34
    sigma : float
        width parameter for basis functions

    Returns:
    --------
    M, Mc : numpy arrays
        ``Mc`` is the reconstructed-image basis rho_k(r_i) (~Gaussians),
               corresponds to Z^T in the article.
        ``M``  is the projected basis chi_k(x_i),
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
        ek = (1 - log(k2)) * k2 if k2 else 1.0

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
            # index of the largest component
            lmax = min(int(u2), k2)
            # halfwidth of the important range
            delta = 7 * k  # "7" reproduces the old method accuracy (~1e-8 abs)
            # summation limits: +-delta from the maximum, but within [0, k^2]
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
