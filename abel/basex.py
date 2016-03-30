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
# version 0.62 - 2016-03-07 
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


def basex_transform(data, nbf='auto', basis_dir='./', dr=1.0, verbose=True,
                    direction='inverse'):
    """ 
    This function performs the BASEX (BAsis Set EXpansion) 
    Abel Transform. It works on a "right side" image. I.e., 
    it works on just half of a cylindrically symmetric
    object and ``data[0,0]`` should correspond to a central pixel. 
    To perform a BASEX transorm on 
    a whole image, use ::
    
        abel.Transform(image, method='basex', direction='inverse').transform

    This BASEX implementation only works with images that have an 
    odd-integer width.
    
    Note: only the `direction="inverse"` transform is currently implemented.
    

    Parameters
    ----------
    data : a NxM numpy array
        The image to be inverse transformed. The width (M) must be odd and ``data[:,0]``
        should correspond to the central column of the image. 
    nbf : str or list
        number of basis functions. If ``nbf='auto'``, it is set to ``(n//2 + 1)``.
        *This is what you should always use*,
        since this BASEX implementation does not work reliably in other situations!
        In the future, you could use
        list, in format [nbf_vert, nbf_horz]
    basis_dir : str
        path to the directory for saving / loading the basis set coefficients.
        If None, the basis set will not be saved to disk.
    dr : float
        size of one pixel in the radial direction. 
        This only affects the absolute scaling of the transformed image.
    verbose : boolean
        Determines if statements should be printed.
    direction : str
        The type of Abel transform to be performed.
        Currently only accepts value ``'inverse'``.


    Returns
    -------
    recon : NxM numpy array
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
        raise ValueError('Wrong input shape for \
                            data {0}, should be (N1, N2) \
                            or (1, N), not (N, 1)'.format(data.shape))
    else:
        data_ndim = 2

    full_image = np.hstack((np.fliplr(data), data[:, 1:]))
    n = full_image.shape

    # load the basis sets:
    M_vert, M_horz, Mc_vert, \
        Mc_horz, vert_left, horz_right = get_bs_basex_cached(
            n_vert=n[0], n_horz=n[1], nbf=nbf, basis_dir=basis_dir,
            verbose=verbose)
            
    # Do the actual transform:
    recon = basex_core_transform(full_image, M_vert, M_horz,
                                  Mc_vert, Mc_horz, vert_left, horz_right, dr)
                                  
    if data_ndim == 1:  # taking the middle row, since the rest are zeroes
        recon = recon[recon.shape[0] - recon.shape[0]//2 - 1]
    if h == 1:
        return recon[w-1:]
    else:
        return recon[:, w-1:]


def basex_core_transform(rawdata, M_vert, M_horz, Mc_vert,
                         Mc_horz, vert_left, horz_right, dr=1.0):
    """
    This is the internal function
    that does the actual BASEX transform. It requires 
    that the matrices of basis set coefficients be passed. 
    

    Parameters
    ----------
    rawdata : NxM numpy array
        the raw image.
    M_vert_etc. : Numpy arrays
        2D arrays given by the basis set calculation function
    dr : float
        pixel size. This only affects the absolute scaling of the output.


    Returns
    -------
    IM : NxM numpy array
        The abel-transformed image, a slice of the 3D distribution
    """

    # Reconstructing image  - This is where the magic happens
    Ci = scipy.dot(scipy.dot(vert_left, rawdata), horz_right) # previously: vert_left.dot(rawdata).dot(horz_right)

    # use an heuristic scaling factor to match the analytical abel transform
    # For more info see https://github.com/PyAbel/PyAbel/issues/4
    MAGIC_NUMBER = 1.1122244156826457
    Ci *= MAGIC_NUMBER/dr
    IM = scipy.dot(scipy.dot(Mc_vert, Ci), Mc_horz.T)    # Previously: Mc_vert.dot(Ci).dot(Mc_horz.T)
    # P = dot(dot(Mc,Ci),M.T) # This calculates the projection,
    # which should recreate the original image
    return IM


def _get_left_right_matrices(M_vert, M_horz, Mc_vert, Mc_horz):
    """ An internal helper function for no-up/down-asymmetry BASEX:
        given basis sets  M_vert, M_horz, Mc_vert, Mc_horz,
        return M_left and M_right matrices
    """

    nbf_vert, nbf_horz = np.shape(M_vert)[1], np.shape(M_horz)[1]
    q_vert, q_horz = 0, 0  # No Tikhonov regularization
    E_vert, E_horz = np.identity(nbf_vert)*q_vert, np.identity(nbf_horz)*q_horz

    vert_left  = scipy.dot( inv(scipy.dot(Mc_vert.T, Mc_vert) + E_vert),  Mc_vert.T)
    horz_right = scipy.dot( M_horz, inv(scipy.dot(M_horz.T, M_horz) + E_horz) )
    
    # previously: 
    # vert_left = inv(Mc_vert.T.dot(Mc_vert) + E_vert).dot(Mc_vert.T)
    # horz_right = M_horz.dot(inv(M_horz.T.dot(M_horz) + E_horz))
    

    return vert_left, horz_right


def _nbf_default(n_vert, n_horz, nbf):
    """ An internal helper function for the asymmetric case
        to check that nbf = n//2 + 1 and print a warning otherwise
    """
    if nbf == 'auto':
        # nbf_vert = n_vert (if relevant)
        # nbf_horz = n_horz//2 + 1
        nbf = [n_vert, n_horz//2 + 1]
    elif isinstance(nbf, (int, long)):
        if nbf != n_horz//2 + 1:
            print('Warning: the number of basis functions \
                    nbf = {} != (n//2 + 1)  = {}\n'.format(
                        nbf, n_horz//2 + 1))
            print('This behaviour is currently not tested \
                    and should not be used \
                    unless you know exactly what you are doing. \
                    Setting nbf="auto" is best for now.')
        nbf = [nbf]*2  # Setting identical number of vert and horz functions
    elif isinstance(nbf, (list, tuple)):
        if nbf[-1] != n_horz//2 + 1:
            print('Warning: the number of basis functions \
                    nbf = {} != (n//2 + 1) = {}\n'.format(
                        nbf[-1], n_horz//2 + 1))
            print('This behaviour is currently not tested \
                    and should not be used \
                    unless you know exactly what you are doing. \
                    Setting nbf="auto" is best for now.')
        if len(nbf) < 2:
            nbf = nbf*2
            # In case user inputs [nbf] instead of [nbf_vert, nbf_horz]
    else:
        raise ValueError('nbf must be set to "auto" or an integer or a list')
    return nbf


def get_bs_basex_cached(n_vert, n_horz,
                        nbf='auto', basis_dir='.', verbose=False):
    """
    Internal function.

    Gets BASEX basis sets, using the disk as a cache
    (i.e. load from disk if they exist,
    if not calculate them and save a copy on disk).
    To prevent saving the basis sets to disk, set ``basis_dir=None``.

    Parameters
    ----------
    n_vert, n_horz : int
        Abel inverse transform will be performed on a
        ``n_vert x n_horz`` area of the image
    nbf : int or list
        number of basis functions. If ``nbf='auto'``, 
        ``n_horz`` is set to ``(n//2 + 1)``.
    basis_dir : str
        path to the directory for saving / loading
        the basis set coefficients.
        If None, the basis sets will not be saved to disk.

    Returns
    -------
    M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right: numpy arrays
        the matrices that compose the basis set.
    """

    # Sanitize nbf
    nbf = _nbf_default(n_vert, n_horz, nbf)
    nbf_vert, nbf_horz = nbf[0], nbf[1]

    basis_name = "basex_basis_{}_{}_{}_{}.npy".format(
                        n_vert, n_horz, nbf_vert, nbf_horz)

    M_horz = None
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file):  # Use existing basis set
            if verbose:
                print('Loading basis sets...           ')
                # saved as a .npy file
            try:
                M_vert, M_horz, Mc_vert, Mc_horz, vert_left, \
                    horz_right, M_version = np.load(path_to_basis_file)
            except ValueError:
                raise print("Cached basis file incompatible. \
                    Please delete the saved basis file and try again.")

    if M_horz is None:  # generate the basis set
        if verbose:
            print('A suitable basis set was not found.',
                  'A new basis set will be generated.',
                  'This may take a few minutes. ', end='')
            if basis_dir is not None:
                print('But don\'t worry, \
                       it will be saved to disk for future use.\n')
            else:
                print(' ')

        M_vert, M_horz, Mc_vert, Mc_horz = _bs_basex(
            n_vert, n_horz, nbf_vert, nbf_horz, verbose=verbose)
        vert_left, horz_right = _get_left_right_matrices(
            M_vert, M_horz, Mc_vert, Mc_horz)

        if basis_dir is not None:
            np.save(path_to_basis_file,
                    (M_vert, M_horz, Mc_vert, Mc_horz, vert_left,
                        horz_right,  np.array(__version__)))
            if verbose:
                print('Basis set saved for later use to,')
                print(' '*10 + '{}'.format(path_to_basis_file))
    return M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right

MAX_BASIS_SET_OFFSET = 4000


def _bs_basex(n_vert=1001, n_horz=501,
              nbf_vert=1001, nbf_horz=251, verbose=True):
    """
    Generates basis sets for the BASEX method
    without assuming up/down symmetry.

    Parameters:
    -----------
    n_vert : integer
        Vertical dimensions of the image in pixels.
    n_horz : integer
        Horizontal dimensions of the image in pixels.
        Must be odd. See https://github.com/PyAbel/PyAbel/issues/34
    nbf_vert : integer
        Number of basis functions in the z-direction.
        Must be less than or equal (default) to n_vert
    nbf_horz : integer
        Number of basis functions in the x-direction.
        Must be less than or equal (default) to n_horz//2 + 1

    Returns:
    --------
      M_vert, M_horz, Mc_vert, Mc_horz : numpy arrays
    """

    if n_horz % 2 == 0:
        raise ValueError('The horizontal dimensions of the image (n_horz) \
                          must be odd.')

    if nbf_horz > n_horz//2 + 1:
        raise ValueError('The number of horizontal basis functions (nbf_horz) \
                          cannot be greater than n_horz//2 + 1')

    if nbf_vert > n_vert:
        raise ValueError('The number of vertical basis functions (nbf_vert) \
                          cannot be greater than \
                          the number of vertical pixels (n_vert).')

    Rm_h = n_horz//2 + 1

    I_h = np.arange(1, n_horz + 1)

    R2_h = (I_h - Rm_h)**2
    M_horz = np.zeros((n_horz, nbf_horz))
    Mc_horz = np.zeros((n_horz, nbf_horz))

    M_horz[:, 0] = 2*np.exp(-R2_h)
    Mc_horz[:, 0] = np.exp(-R2_h)

    gammaln_0o5 = gammaln(0.5)

    if verbose:
        print('Generating horizontal BASEX basis sets for \
               n_horz = {}, nbf_vert = {}:\n'.format(n_horz, nbf_vert))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf_horz)*32 - MAX_BASIS_SET_OFFSET,
                    MAX_BASIS_SET_OFFSET)
    for k in range(1, nbf_horz):
        k2 = k*k  # so we don't recalculate it all the time
        log_k2 = log(k2)
        angn = exp(k2 * (1 - log_k2) +
                   gammaln(k2 + 0.5) - gammaln_0o5)
        M_horz[Rm_h-1, k] = 2 * angn
        for l in range(1, n_horz - Rm_h + 1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 * (1 + log(l2/k2)) - l2)

            Mc_horz[Rm_h - 1 + l, k] = val  # All rows below center
            Mc_horz[Rm_h - 1 - l, k] = val  # All rows above center

            aux = val + angn * Mc_horz[Rm_h - 1 + l, 0]

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

            M_horz[Rm_h - 1 + l, k] = aux  # All rows below center
            M_horz[Rm_h - 1 - l, k] = aux  # All rows above center

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))
    """
    # Axial functions
    """
    Z2_h = np.arange(0, n_vert)**2
    M_vert = np.zeros((n_vert, nbf_vert))
    Mc_vert = np.zeros((n_vert, nbf_vert))
    Mc_vert[:, 0] = np.exp(-Z2_h)
    if verbose:
        print('Generating vertical BASEX basis sets for n_vert = {}, \
            nbf_vert = {}:\n'.format(n_vert, nbf_vert))
        sys.stdout.flush()

    k = np.arange(1, nbf_vert)
    k2 = (k*k)[None, :]
    l = np.arange(1, n_vert)
    l2 = (l*l)[:, None]
    Mc_vert[1:, 1:] = np.exp(k2 * (1 + np.log(l2/k2)) - l2)

    if verbose:
        print("...{}".format(k+1))
    return M_vert, M_horz, Mc_vert, Mc_horz
