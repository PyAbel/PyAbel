# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time
import os.path
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


def basex_transform(data, nbf='auto', reg=0.0, bs_correction=False,
                    basis_dir='./', dr=1.0,
                    verbose=True, direction='inverse'):
    """
    This function performs the BASEX (BAsis Set EXpansion)
    Abel Transform. It works on a "right side" image. I.e.,
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
    nbf : str or int
        number of basis functions. If ``nbf='auto'``, it is set to ``n``.
        *This is what you should always use*, since this BASEX implementation
        does not work reliably in other situations!
        In the future, you could use other numbers
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
    Ai = get_bs_basex_cached(n, nbf=nbf, reg=reg, bs_correction=bs_correction,
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
        The abel-transformed image, a slice of the 3D distribution
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
    #     corresponds to A^T Z in the article
    # reg corresponds to q_1^2 in the article

    if reg == 0.0:
        Ai = scipy.dot(inv(M.T), Mc.T)  # (this works only for nbf == n)
    else:
        nbf = np.shape(M)[1]
        # square of Tikhonov matrix
        E = np.diag([reg] * nbf)
        # regularized inverse of basis projection
        R = scipy.dot(M, inv(scipy.dot(M.T, M) + E))
        # {expansion coefficients} = projection . R
        # image = {expansion coefficients} . {image basis}
        # so: image = projection . (R . {image basis})
        #     image = projection . Ai
        #     Ai = R . {image basis}
        # Ai is the matrix of the inverse Abel transform
        Ai = scipy.dot(R, Mc.T)

    # use an heuristic scaling factor to match the analytical abel transform
    # For more info see https://github.com/PyAbel/PyAbel/issues/4
    MAGIC_NUMBER = 1.1122244156826457
    Ai *= MAGIC_NUMBER

    return Ai


def _nbf_default(n, nbf):
    """ An internal helper function for the asymmetric case
        to check that nbf = n and print a warning otherwise
    """
    if nbf == 'auto':
        nbf = n
    elif isinstance(nbf, (int, long)):
        if nbf != n:
            print('Warning: the number of basis functions '
                  'nbf = {} != n  = {}\n'.format(nbf, n))
            print('This behaviour is currently not tested '
                  'and should not be used '
                  'unless you know exactly what you are doing. '
                  'Setting nbf="auto" is best for now.')
    else:
        raise ValueError('nbf must be set to "auto" or an integer')
    return nbf


# Cached matrices and their parameters
_prm = None  # [n, nbf, bs_correction]
_M = None    # [M, Mc]
_reg = None  # reg
_Ai = None   # Ai

def get_bs_basex_cached(n, nbf='auto', reg=0.0, bs_correction=False,
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
    nbf : int
        number of basis functions. If ``nbf='auto'``,
        ``n`` is set to ``n``.
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

    # Sanitize nbf
    nbf = _nbf_default(n, nbf)

    M = None
    # Check whether basis for these parameters is already loaded
    if _prm == [n, nbf, bs_correction]:
        if verbose:
            print('Using memory-cached basis sets')
        M, Mc = _M
    else:  # try to load basis
        basis_name = 'basex_basis_{}_{}.npy'.format(n, nbf)

        if basis_dir is not None:
            path_to_basis_file = os.path.join(basis_dir, basis_name)
            if os.path.exists(path_to_basis_file):  # Use existing basis set
                if verbose:
                    print('Loading basis sets...')
                    # saved as a .npy file
                try:
                    M, Mc, M_version = np.load(path_to_basis_file)
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

            M, Mc = _bs_basex(n, nbf, verbose=verbose)

            if basis_dir is not None:
                np.save(path_to_basis_file,
                        (M, Mc, np.array(__version__)))
                if verbose:
                    print('Basis set saved for later use to')
                    print('  {}'.format(path_to_basis_file))

        # Apply basis correction
        if bs_correction:
            # This is a dirty hack!
            # See https://github.com/PyAbel/PyAbel/issues/230
            l = min(nbf, 5)  # modifying at most 5 first points (what fits)
            # image basis function k = 0
            Mc[:l, 0] = [1.27, 0.19, -0.025, -0.015, -0.007][:l]
            # its projection
            M[:l, 0] = [1.65, 0.18, -0.15, -0.09, -0.04][:l]

        _prm = [n, nbf, bs_correction]
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


MAX_BASIS_SET_OFFSET = 4000

def _bs_basex(n=251, nbf=251, verbose=True):
    """
    Generates basis sets for the BASEX method
    without assuming up/down symmetry.

    Parameters:
    -----------
    n : integer
        horizontal dimensions of the half-width image in pixels.
        Must inglude the axial pixel.
        See https://github.com/PyAbel/PyAbel/issues/34
    nbf : integer
        number of basis functions in the x direction.
        Must be less than or equal (default) to n

    Returns:
    --------
      M, Mc : numpy arrays
    """

    # Mc is the reconstructed-image basis (Gaussians),
    #    corresponds to Z^T in the article
    # M  is the basis projection,
    #    corresponds to X^T in the article

    if nbf > n:
        raise ValueError('The number of horizontal basis functions (nbf) '
                         'cannot be greater than n')

    nf = 2 * n - 1  # full width

    Rm_h = n

    I_h = np.arange(1, nf + 1)

    R2_h = (I_h - Rm_h)**2
    M = np.zeros((nf, nbf))
    Mc = np.zeros((nf, nbf))

    M[:, 0] = 2*np.exp(-R2_h)
    Mc[:, 0] = np.exp(-R2_h)

    gammaln_0o5 = gammaln(0.5)

    if verbose:
        print('Generating horizontal BASEX basis sets for '
              'n = {}:\n'.format(n))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf)*32 - MAX_BASIS_SET_OFFSET,
                    MAX_BASIS_SET_OFFSET)
    for k in range(1, nbf):
        k2 = k*k  # so we don't recalculate it all the time
        log_k2 = log(k2)
        angn = exp(k2 * (1 - log_k2) +
                   gammaln(k2 + 0.5) - gammaln_0o5)
        M[Rm_h-1, k] = 2 * angn
        for l in range(1, nf - Rm_h + 1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 * (1 + log(l2/k2)) - l2)

            Mc[Rm_h - 1 + l, k] = val  # All rows below center
            # Mc[Rm_h - 1 - l, k] = val  # All rows above center

            aux = val + angn * Mc[Rm_h - 1 + l, 0]

            p = np.arange(max(1, l2 - delta[k]),
                          min(k2 - 1,  l2 + delta[k]) + 1)
            """
            We use here the fact that for p, k real and positive
            np.log(np.arange(p, k)).sum() == gammaln(k) - gammaln(p)
            where gammaln is scipy.misc.gammaln
            (i.e. the log of the Gamma function)
            The following line corresponds to the vectorized third
            loop of the original BASIS2.m matlab file.
            """
            aux += np.exp(
                k2 - l2 - k2*log_k2 + p*log_l2 +
                gammaln(k2 + 1) - gammaln(p + 1) + gammaln(k2 - p + 0.5) -
                gammaln_0o5 - gammaln(k2 - p + 1)).sum()
            # End of vectorized third loop
            aux *= 2

            M[Rm_h - 1 + l, k] = aux  # All rows below center
            # M[Rm_h - 1 - l, k] = aux  # All rows above center

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print('...{}'.format(k+1))

    # taking only needed halves
    # (all the code above must be modified to work with halves throughout)
    M = M[n-1:, :]
    Mc = Mc[n-1:, :]

    return M, Mc
