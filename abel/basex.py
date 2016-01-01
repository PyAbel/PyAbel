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
from numpy.linalg import inv
from scipy.ndimage import median_filter, gaussian_filter

from ._version import __version__
from .tools import calculate_speeds, center_image, center_image_asym

#############################################################################
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "The Gaussian basis-set expansion Abel transform method"
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler,
# Review of Scientific Instruments 73, 2634 (2002).
#
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


def BASEX(data, center, n, 
        nbf='auto',  basis_dir='./', calc_speeds=False, vertical_symmetry=True, dr=1.0, verbose=True):

    """ This function that centers the image, performs the BASEX transform (loads or generates basis sets), 
        and (optionally) calculates the radial integration of the image (calc_speeds)

    Parameters:
    -----------
      - data:  a NxM numpy array
            If data is smaller than the size of the basis set, zeros will be padded on the edges.
      - n:  * odd integer -
                Abel inverse transform will be performed on a `n x n` area of the image
            * list in format [n_vert, n_horz] - 
                Abel inverse transform will be performed on a `n[0] x n[1]` area of the image
      - nbf: * integer - 
                number of basis functions. If nbf='auto', it is set to (n//2 + 1).
             * list in format [nbf_vert, nbf_horz] -
                If nbf='auto', it is set to [n_vert, n_horz//2 + 1]
      - center: * integer - 
                    the center column of the image
                * tuple (x,y) -
                    the center of the image in (x,y) format
      - basis_dir: string
            path to the directory for saving / loading the basis set coefficients.
            If None, the basis set will not be saved to disk. 
      - dr: float
            size of one pixel in the radial direction
      - calc_speeds: True/False
            determines if the speed distribution should be calculated
      - vertical_symmetry: True/False
            determines if the data has up/down symmetry 
      - verbose: True/False
            Set to True to see more output for debugging

    Returns:
       if calc_speeds=False: the processed image
       if calc_speeds=True:  the processes images, arrays with the calculated speeds

    """
    # make dimension-of-rawdata into list to account for rectangular n
    # Format of n -> n = [n_vert, n_horz]
    if type(n) is not list: 
        if type(n) is int: n = [ n ] 
        else: n = list(n)

    # duplicate elements of n if n is single-valued (rawdata is square)
    if len(n) < 2: n = n*2

    # make sure n_horz is odd
    n[1] = 2 * (n[1] // 2) + 1 

    # make sure that the data is the right shape (1D must be converted to 2D)
    data = np.atleast_2d(data) # if passed a 1D array convert it to 2D
    if data.shape[0] == 1:
        data_ndim = 1
    elif data.shape[1] == 1:
        raise ValueError('Wrong input shape for data {0}, should be  (N1, N2) or (1, N), not (N, 1)'.format(data.shape))
    else:
        data_ndim = 2

    if vertical_symmetry:
        image = center_image(data, center=center, n=n[1], ndim=data_ndim)

        if verbose:
            t1 = time()

        M, Mc, M_left, M_right = get_bs_basex_cached(n[1], nbf, basis_dir, verbose)

        if verbose:
            print('{:.2f} seconds'.format((time() - t1)))

        # Do the actual transform
        if verbose:
            print('Reconstructing image...         ')
            t1 = time()

        recon = basex_transform(image, M, Mc, M_left, M_right, dr)

        if verbose:
            print('%.2f seconds' % (time() - t1))

        if data_ndim == 1:
            recon = recon[0, :] # taking one row, since they are all the same anyway

        if calc_speeds:
            if verbose:
                print('Generating speed distribution...')
                t1 = time()

            speeds = calculate_speeds(recon)

            if verbose:
                print('%.2f seconds' % (time() - t1))
            return recon, speeds
        else:
            return recon

    else: # No vertical symmetry
        if type(center) == tuple: cx, cy = center
        elif type(center) == int: cx = center
        else: raise ValueError("Center specified incorrectly. Must be tuple (x,y) or integer (column)")

        image = center_image_asym(data, center_column = cx, n_vert = n[0], n_horz = n[1], verbose = verbose)

        if verbose:
            t1 = time()

        M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right = get_bs_basex_cached_asym(n_vert = n[0], n_horz = n[1], nbf = nbf, basis_dir = basis_dir, verbose = verbose)

        if verbose:
            print('{:.2f} seconds'.format((time() - t1)))

        #Do the actual transform
        if verbose:
            print('Reconstructing image...         ')
            t1 = time()

        recon = basex_transform_asym(image, M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right, dr)

        if verbose:
            print('%.2f seconds' % (time() - t1))

        if data_ndim == 1:
            recon = recon[recon.shape[0] - recon.shape[0]//2 - 1] # taking the middle row, since the rest are zeroes

        # -------------------------------------------------
        # asymmetric speeds calculation not implemented yet
        # -------------------------------------------------
        # if calc_speeds:
        #     if verbose:
        #         print('Generating speed distribution...')
        #         t1 = time()

        #     speeds = calculate_speeds(recon, n)

        #     if verbose:
        #         print('%.2f seconds' % (time() - t1))
        #     return recon, speeds
        # else:
        #     return recon

        return recon




def basex_transform(rawdata, M, Mc, M_left, M_right, dr=1.0):
    """ This is the internal function that does the actual BASEX transform

     Parameters
     ----------
      - rawdata: a NxN ndarray of the raw image.
      - M, Mc, M_left, M_right: 2D arrays given given by the basis set calculation function
      - dr: float: pixel size

     Returns
     -------
      IM: The abel-transformed image, a slice of the 3D distribution
    """

    Ci = M_left.dot(rawdata).dot(M_right)
    # P = dot(dot(Mc,Ci),M.T) # This calculates the projection, which should recreate the original image
    IM = Mc.dot(Ci).dot(Mc.T)

    # use an heuristic scaling factor to match the analytical abel transform
    # see https://github.com/PyAbel/PyAbel/issues/4
    # When nbf = n//2, we need the MAGIC_NUMBER = 8.053
    # When nbf = n//2 + 1, see below,
    MAGIC_NUMBER = 1.1122244156826457 
    # MAGIC_NUMBER = 8.053 
    IM *= MAGIC_NUMBER/dr

    return IM

def basex_transform_asym(rawdata, M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right, dr=1.0):
    """ This is the internal function that does the actual BASEX transform for the no-up/down-symmetry case.
     Parameters
     ----------
      - rawdata: 
            a N_vert x N_horz numpy array of the raw image.
      - M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right: 
            2D arrays given by the basis set calculation function
      - dr: float: pixel size

     Returns
     -------
      IM: The abel-transformed image, a slice of the 3D distribution
    """
    ### Reconstructing image  - This is where the magic happens ###
    Ci = vert_left.dot(rawdata).dot(horz_right)

    # use an heuristic scaling factor to match the analytical abel transform
    # For more info see https://github.com/PyAbel/PyAbel/issues/4
    MAGIC_NUMBER = 1.1122244156826457
    Ci *= MAGIC_NUMBER/dr 

    IM = Mc_vert.dot(Ci).dot(Mc_horz.T)
    # P = dot(dot(Mc,Ci),M.T) # This calculates the projection, which should recreate the original image

    return IM


def _get_left_right_matrices(M, Mc):
    """ An internal helper function for BASEX:
        given basis sets  M, Mc return M_left and M_right matrices 
        M_left and M_right are the A and B matrices (respectively) 
        from Equation 13 of the Dribinski et al. paper."""

    M_left = inv(Mc.T.dot(Mc)).dot(Mc.T) 
    q = 1  # q is the regularization factor from Equation 8 of the Dribinski paper.
           # Changing this from 1 may improve performance in some situations?
    nbf = M.shape[1]        # number of basis functions
    E = np.identity(nbf)*q  # Creating diagonal matrix for regularization. (?)
    M_right = M.dot(inv((M.T.dot(M) + E)))
    return M_left, M_right

def _get_left_right_matrices_asym(M_vert, M_horz, Mc_vert, Mc_horz): 
    """ An internal helper function for no-up/down-asymmetry BASEX:
        given basis sets  M_vert, M_horz, Mc_vert, Mc_horz,
        return M_left and M_right matrices 
    """

    nbf_vert, nbf_horz = np.shape(M_vert)[1], np.shape(M_horz)[1]
    
    q_vert, q_horz = 0,0 # No Tikhonov regularization
    E_vert, E_horz = np.identity(nbf_vert)*q_vert, np.identity(nbf_horz)*q_horz

    vert_left = inv(Mc_vert.T.dot(Mc_vert) + E_vert).dot(Mc_vert.T)
    horz_right = M_horz.dot(inv(M_horz.T.dot(M_horz) + E_horz))

    return vert_left, horz_right

def _nbf_default(n, nbf):
    """ An internal helper function to check that nbf = n//2 + 1 and print a warning
    otherwise """
    if nbf == 'auto':
        nbf = n//2 + 1
    else:
        if nbf != n//2 +1:
            print('Warning: the number of basis functions nbf = {} != (n//2 + 1) = {}\n'.format(n, nbf),
                    '    This behaviour is currently not tested and should not be used\
                    unless you know exactly what you are doing. Setting nbf="auto" is best for now.')

    return nbf

def _nbf_default_asym(n_vert, n_horz, nbf):
    """ An internal helper function for the asymmetric case to check that nbf = n//2 + 1 and print a warning otherwise """
    if nbf == 'auto':
        # nbf_vert = n_vert (if relevant)
        # nbf_horz = n_horz//2 + 1
        nbf = [n_vert, n_horz//2 + 1]
    elif type(nbf) == int:
        if nbf != n_horz//2 +1:
            print('Warning: the number of basis functions nbf = {} != (n//2 + 1)  = {}\n'.format(nbf, n_horz//2 +1),
                    '    This behaviour is currently not tested and should not be used\
                    unless you know exactly what you are doing. Setting nbf="auto" is best for now.')
        nbf = [nbf]*2 # Setting identical number of vert and horz functions
    elif isinstance(nbf, (list, tuple)):
        if nbf[-1] != n_horz//2 +1:
            print('Warning: the number of basis functions nbf = {} != (n//2 + 1) = {}\n'.format(nbf[-1], n_horz//2 +1),
                    '    This behaviour is currently not tested and should not be used\
                    unless you know exactly what you are doing. Setting nbf="auto" is best for now.')
        if len(nbf) < 2: nbf = nbf*2 # In case user inputs [nbf] instead of [nbf_vert, nbf_horz]
    else:
        raise ValueError('nbf must be set to "auto" or an integer or a list')
    return nbf


def get_bs_basex_cached(n, nbf='auto', basis_dir='.', verbose=False):
    """
    Internal function.

    Get basis sets, using the disk as a cache 
    (i.e. load from disk if they exist, if not calculate them and save a copy on disk).

    Parameters:
    -----------
      - n : odd integer: Abel inverse transform will be performed on a `n x n`
        area of the image
      - nbf: integer: number of basis functions. If nbf='auto', it is set to (n//2 + 1).
      - basis_dir : path to the directory for saving / loading the basis set coefficients.
                    If None, the basis sets will not be saved to disk. 
    """
    # Sanitizing nbf
    nbf = _nbf_default(n, nbf)

    basis_name = "basex_basis_{}_{}.npy".format(n, nbf)

    M = None
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file):
            if verbose:
                print('Loading basis sets...           ')
                # saved as a .npy file
            try:
                M, Mc, M_left, M_right, M_version  = np.load(path_to_basis_file)
            except ValueError:
                try: # falling back to the legacy basis set file format
                    M_left, M_right, M, Mc = np.load(path_to_basis_file)
                except:
                    raise
            except:
                raise

    if M is None:
        if verbose:
            print('A suitable basis set was not found.',
                  'A new basis set will be generated.',
                  'This may take a few minutes. ', end='')
            if basis_dir is not None:
                print('But don\'t worry, it will be saved to disk for future use.\n')
            else:
                print(' ')

        M, Mc = generate_basis_sets(n, nbf, verbose=verbose)
        M_left, M_right = _get_left_right_matrices(M, Mc)

        if basis_dir is not None:
            np.save(path_to_basis_file,
                    (M, Mc, M_left, M_right,  np.array(__version__)))
            if verbose:
                print('Basis set saved for later use to,')
                print(' '*10 + '{}'.format(path_to_basis_file))
    return M, Mc, M_left, M_right


def get_bs_basex_cached_asym(n_vert, n_horz, nbf='auto', basis_dir='.', verbose=False):
    """
    Internal function.

    Get up/down-asymmetric basis sets, using the disk as a cache 
    (i.e. load from disk if they exist, if not calculate them and save a copy on disk).

    Parameters:
    -----------
      - n_vert, n_horz:
            integer: Abel inverse transform will be performed on a `n_vert x n_horz` area of the image
      - nbf: 
            integer or list: number of basis functions. If nbf='auto', n_horz is set to (n//2 + 1).
      - basis_dir : path to the directory for saving / loading the basis set coefficients. If None, the basis sets will not be saved to disk. 

    Returns:
    --------
      - M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right: numpy arrays
    """

    # Sanitize nbf
    nbf = _nbf_default_asym(n_vert, n_horz, nbf)
    nbf_vert, nbf_horz = nbf[0], nbf[1]

    basis_name = "basex_asymm_basis_{}_{}_{}_{}.npy".format(n_vert, n_horz, nbf_vert, nbf_horz)

    M_horz = None
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file): # Use existing basis set
            if verbose:
                print('Loading basis sets...           ')
                # saved as a .npy file
            try:
                M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right, M_version  = np.load(path_to_basis_file)
            except ValueError:
                raise print("Cached basis file incompatible. Please delete the saved basis file and try again.")

    if M_horz is None: # generate the basis set
        if verbose:
            print('A suitable basis set was not found.',
                  'A new basis set will be generated.',
                  'This may take a few minutes. ', end='')
            if basis_dir is not None:
                print('But don\'t worry, it will be saved to disk for future use.\n')
            else:
                print(' ')

        M_vert, M_horz, Mc_vert, Mc_horz  = _bs_basex_asym(n_vert, n_horz, nbf_vert, nbf_horz, verbose=verbose)
        vert_left, horz_right = _get_left_right_matrices_asym(M_vert, M_horz, Mc_vert, Mc_horz)

        if basis_dir is not None:
            np.save(path_to_basis_file,
                    (M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right,  np.array(__version__)))
            if verbose:
                print('Basis set saved for later use to,')
                print(' '*10 + '{}'.format(path_to_basis_file))

    return M_vert, M_horz, Mc_vert, Mc_horz, vert_left, horz_right


MAX_BASIS_SET_OFFSET = 4000


def generate_basis_sets(n, nbf='auto', verbose=False):
    """ 
    Generate the basis set for the BASEX method. 

    This function was adapted from the matlab script provided by
    the Reisler group: BASIS2.m, with some optimizations.
    
    See original Matlab program here:
    https://github.com/PyAbel/PyAbelLegacy/tree/master/Matlab
    
    Typically, the number of basis functions will be (n//2 + 1)
    so that each pixel in the image is represented by its own basis function.

    Parameters:
    -----------
      n : integer : size of the basis set (pixels)
      nbf: integer: number of basis functions. If nbf='auto', it is set to (n//2 + 1).
      verbose: bool: print progress 

    Returns:
    --------
      M, Mc : np.array 2D:  basis sets
    """
    if n % 2 == 0:
        raise ValueError('The n parameter must be odd (more or less sure about it).')

    nbf = _nbf_default(n, nbf)

    if nbf > n//2 + 1:
        raise ValueError('The number of basis functions nbf cannot be larger then the number of points n!')

    Rm = n//2 + 1

    I = np.arange(1, n+1)

    R2 = (I - Rm)**2
    # R = I - Rm
    M = np.zeros((n, nbf))
    Mc = np.zeros((n, nbf))

    M[:,0] = 2*np.exp(-R2)
    Mc[:,0] = np.exp(-R2)

    gammaln_0o5 = gammaln(0.5) 

    if verbose:
        print('Generating BASEX basis sets for n = {}, nbf = {}:\n'.format(n, nbf))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficients
    delta = np.fmax(np.arange(nbf)*32 - MAX_BASIS_SET_OFFSET, MAX_BASIS_SET_OFFSET)
    for k in range(1, nbf):
        k2 = k*k # so we don't recalculate it all the time
        log_k2 = log(k2) 
        angn = exp(
                    k2 - 2*k2*log(k) +
                    #np.log(np.arange(0.5, k2 + 0.5)).sum() # original version
                    gammaln(k2 + 0.5) - gammaln_0o5  # optimized version
                    )
        M[Rm-1, k] =  2*angn

        for l in range(1, n-Rm+1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 - l2 + 2*k2*log((1.0*l)/k))
            Mc[l-1+Rm, k] = val
            Mc[Rm-l-1, k] = val

            aux = val + angn*Mc[l+Rm-1, 0]

            p = np.arange(max(1, l2 - delta[k]), min(k2 - 1,  l2 + delta[k]) + 1)

            # We use here the fact that for p, k real and positive
            #
            #  np.log(np.arange(p, k)).sum() == gammaln(k) - gammaln(p) 
            #
            # where gammaln is scipy.misc.gammaln (i.e. the log of the Gamma function)
            #
            # The following line corresponds to the vectorized third
            # loop of the original BASIS2.m matlab file.

            aux += np.exp(k2 - l2 - k2*log_k2 + p*log_l2
                      + gammaln(k2+1) - gammaln(p+1) 
                      + gammaln(k2 - p + 0.5) - gammaln_0o5
                      - gammaln(k2 - p + 1)
                      ).sum()

            # End of vectorized third loop

            aux *= 2

            M[l+Rm-1, k] = aux
            M[Rm-l-1, k] = aux

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))

    return M, Mc

def _bs_basex_asym(n_vert=1001, n_horz = 501, 
                                    nbf_vert = 1001, nbf_horz = 251, 
                                    verbose=True):    
    """ 
    Generates basis sets for the BASEX method without assuming up/down symmetry. 

    Parameters:
    -----------
      n_vert : integer : Vertical dimensions of the image in pixels. 
      n_horz : integer : Horizontal dimensions of the image in pixels. Must be odd. See https://github.com/PyAbel/PyAbel/issues/34
      nbf_vert : integer : Number of basis functions in the z-direction. Must be less than or equal (default) to n_vert 
      nbf_horz: integer: Number of basis functions in the x-direction. Must be less than or equal (default) to n_horz//2 + 1

    Returns:
    --------
      M_vert, M_horz, Mc_vert, Mc_horz : numpy arrays
    """

    if n_horz % 2 == 0:
        raise ValueError('The horizontal dimensions of the image (n_horz) must be odd.')

    if nbf_horz > n_horz//2 + 1:
        raise ValueError('The number of horizontal basis functions (nbf_horz) cannot be greater than n_horz//2 + 1')

    if nbf_vert > n_vert:
        raise ValueError('The number of vertical basis functions (nbf_vert) cannot be greater than the number of vertical pixels (n_vert).')

    Rm_h = n_horz//2 + 1

    I_h = np.arange(1, n_horz + 1)

    R2_h = (I_h - Rm_h)**2
    M_horz = np.zeros((n_horz, nbf_horz))
    Mc_horz = np.zeros((n_horz, nbf_horz))

    M_horz[:,0] = 2*np.exp(-R2_h)
    Mc_horz[:,0] = np.exp(-R2_h)

    gammaln_0o5 = gammaln(0.5) 

    if verbose:
        print('Generating horizontal BASEX basis sets for n_horz = {}, nbf_horz = {}:\n'.format(n_horz, nbf_horz))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf_horz)*32 - MAX_BASIS_SET_OFFSET, MAX_BASIS_SET_OFFSET) 
    for k in range(1, nbf_horz):
        k2 = k*k # so we don't recalculate it all the time
        log_k2 = log(k2) 
        angn = exp(
                    k2 * (1 - log_k2) + 
                    gammaln(k2 + 0.5) - gammaln_0o5  
                    # old form --> k2 - 2 * k2*log(k) +
                    #              np.log(np.arange(0.5, k2 + 0.5)).sum()
                    )
        M_horz[Rm_h-1, k] =  2 * angn

        for l in range(1, n_horz - Rm_h + 1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 * (1 + log(l2/k2)) - l2) 
            # old form --> val = exp(k2 - l2 + 2 * k2*log((1.0 * l)/k)) 

            Mc_horz[Rm_h - 1 + l, k] = val # All rows below center
            Mc_horz[Rm_h - 1 - l, k] = val # All rows above center

            aux = val + angn * Mc_horz[Rm_h - 1 + l, 0]

            p = np.arange(max(1, l2 - delta[k]), min(k2 - 1,  l2 + delta[k]) + 1)

            # We use here the fact that for p, k real and positive
            #
            #  np.log(np.arange(p, k)).sum() == gammaln(k) - gammaln(p) 
            #
            # where gammaln is scipy.misc.gammaln (i.e. the log of the Gamma function)
            #
            # The following line corresponds to the vectorized third
            # loop of the original BASIS2.m matlab file.

            aux += np.exp(k2 - l2 - k2*log_k2 + p*log_l2
                      + gammaln(k2 + 1) - gammaln(p + 1) 
                      + gammaln(k2 - p + 0.5) - gammaln_0o5
                      - gammaln(k2 - p + 1)
                      ).sum()

            # End of vectorized third loop

            aux *= 2

            M_horz[Rm_h - 1 + l, k] = aux # All rows below center
            M_horz[Rm_h - 1 - l, k] = aux # All rows above center

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))

    ####################################    
    # Axial functions
    ####################################

    Z2_h = np.arange(0, n_vert)**2

    M_vert = np.zeros((n_vert, nbf_vert))
    Mc_vert = np.zeros((n_vert, nbf_vert))

    # M_vert[:,0] = 2*np.exp(-Z2_h)
    Mc_vert[:,0] = np.exp(-Z2_h)

    if verbose:
        print('Generating vertical BASEX basis sets for n_vert = {}, nbf_vert = {}:\n'.format(n_vert, nbf_vert))
        # sys.stdout.write('0')
        sys.stdout.flush()

    # delta_v = np.fmax(np.arange(nbf_vert)*32 - 0, 8000) 

    k = np.arange(1, nbf_vert)
    k2 = (k*k)[None, :]
    l = np.arange(1, n_vert)
    l2 = (l*l)[:, None]
    Mc_vert[1:, 1:] = np.exp(k2 * (1 + np.log(l2/k2)) - l2)

    if verbose:
        print("...{}".format(k+1))

    return M_vert, M_horz, Mc_vert, Mc_horz 
