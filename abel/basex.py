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

from .tools import calculate_speeds, center_image
from .io import parse_matlab


######################################################################
# PyBASEX - A Python BASEX implementation
# Dan Hickstein - University of Colorado Boulder
# danhickstein@gmail.com
#
# This is adapted from the BASEX Matlab code provided by the Reisler group.
#
# Please cite: "The Gaussian basis-set expansion Abel transform method"
# V. Dribinski, A. Ossadtchi, V. A. Mandelshtam, and H. Reisler,
# Review of Scientific Instruments 73, 2634 (2002).
#
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
#
# To-Do list:
#
#   I took all of the linear algebra straight from the Matlab program. It's
#   a little hard to compare with the Rev. Sci. Instrum. paper. It would be
#   nice to clean this up so that it's easier to follow along with the paper.
#
########################################################################




class BASEX(object):

    def __init__(self, n=501, nbf=250, basis_dir='./',
                    use_basis_set=None, verbose=True, calc_speeds=False,
                    pixel_size=1.0, scaling_correction=True):
        """ Initalize the BASEX class, preloading or generating the basis set.

        Parameters:
        -----------
          - N : odd integer: Abel inverse transform will be performed on a `n x n`
            area of the image
          - nbf: integer: number of basis functions ?
          - basis_dir : path to the directory for saving / loading the basis set coefficients.
                        If None, the basis set will not be saved to disk. 
          - use_basis_set: use the basis set stored as a text files, if
                  it provided, the following parameters will be ignored N, nbf, basis_dir
                  The expected format is a string of the form "some_basis_set_{}_1.bsc" where 
                  "{}" will be replaced by "" for the first file and "pr" for the second.
                  Gzip compressed text files are accepted.
          - calc_speeds: determines if the speed distribution should be calculated
          - pixel_size: size of one pixel in the radial direction (optional)
          - scaling_correction: correct the result by a heuristic scaling factor
                            as to be consistent with the analytical inverse Abel transform.
                            This requires to set pixel_size to the correct value.
          - verbose: Set to True to see more output for debugging

        """
        n = 2*(n//2) + 1 # make sure n is odd

        self.verbose = verbose
        self.calc_speeds = calc_speeds
        self.scaling_correction = scaling_correction
        self.pixel_size = pixel_size

        self.n = n
        self.nbf = nbf

        if self.verbose:
            t1 = time()

        basis_name = "basex_basis_{}_{}.npy".format(n, nbf)
        if basis_dir is not None:
            path_to_basis_file = os.path.join(basis_dir, basis_name)
        else:
            path_to_basis_file = None

        if use_basis_set is not None:
            # load the matlab generated basis set
            M, Mc = parse_matlab(use_basis_set)
            left, right = get_left_right_matrices(M, Mc)

            self.n, self.nbf = M.shape # overwrite the provided parameters

        elif basis_dir is not None and os.path.exists(path_to_basis_file):
            # load the basis set generated with this python module,
            # saved as a .npy file
            if self.verbose:
                print('Loading basis sets...           ')
            left, right, M, Mc = np.load(path_to_basis_file)

        else:
            # generate the basis set
            if self.verbose:
                print('A suitable basis set was not found.',
                      'A new basis set will be generated.',
                      'This may take a few minutes. ', end='')
                if basis_dir is not None:
                    print('But don\'t worry, it will be saved to disk for future use.\n')
                else:
                    print(' ')

            M, Mc = generate_basis_sets(n, nbf, verbose=verbose)
            left, right = get_left_right_matrices(M, Mc)

            if basis_dir is not None:
                np.save(path_to_basis_file, (left, right, M, Mc))
                if self.verbose:
                    print('Basis set saved for later use to,')
                    print(' '*10 + '{}'.format(path_to_basis_file))

        self.left, self.right, self.M, self.Mc = left, right, M, Mc



        if self.verbose:
            print('{:.2f} seconds'.format((time() - t1)))


    def _basex_transform(self, rawdata):
        """ This is the core function that does the actual transform, 
            but it's not typically what is called by the user
         INPUTS:
          rawdata: a NxN numpy array of the raw image.
          verbose: Set to True to see more output for debugging
          calc_speeds: determines if the 1D speed distribution should be calculated (takes a little more time)

         RETURNS:
          IM: The abel-transformed image, a slice of the 3D distribution
          speeds: (optional) a array of length=500 of the 1D distribution, integrated over all angles
        """
        left, right, M, Mc = self.left, self.right, self.M, self.Mc


        ### Reconstructing image  - This is where the magic happens ###
        if self.verbose:
            print('Reconstructing image...         ')
            t1 = time()

        Ci = left.dot(rawdata).dot(right)
        # P = dot(dot(Mc,Ci),M.T) # This calculates the projection, which should recreate the original image
        IM = Mc.dot(Ci).dot(Mc.T)

        if self.verbose:
            print('%.2f seconds' % (time() - t1))

        if self.calc_speeds:
            speeds = self.calculate_speeds(IM)
            return IM, speeds
        else:
            return IM


    def _get_scaling_factor(self):

        MAGIC_NUMBER = 8.053   # see https://github.com/PyAbel/PyAbel/issues/4
        return MAGIC_NUMBER/self.pixel_size


    def calculate_speeds(self, IM):

        if self.verbose:
            print('Generating speed distribution...')
            t1 = time()

        speeds = calculate_speeds(IM, self.n)

        if self.verbose:
            print('%.2f seconds' % (time() - t1))

    def __call__(self, data, center,
                             median_size=0, gaussian_blur=0, post_median=0,
                             symmetrize=False):
        """ This is the main function that is called by the user. 
            It center the image, blurs the image (if desired)
            and completes the BASEX transform.

         Inputs:
         data - a NxN numpy array
                If N is smaller than the size of the basis set, zeros will be padded on the edges.
         center - the center of the image in (x,y) format
         median_size - size (in pixels) of the median filter that will be applied to the image before
                       the transform. This is crucial for emiminating hot pixels and other
                       high-frequency sensor noise that would interfere with the transform
         gaussian_blur - the size (in pixels) of the gaussian blur applied before the BASEX tranform.
                         this is another way to blur the image before the transform.
                         It is normally not used, but if you are looking at very broad features
                         in very noisy data and wich to apply an aggressive (large radius) blur
                         (i.e., a blur in excess of a few pixels) then the gaussian blur will
                         provide better results than the median filter.
         post_median - this is the size (in pixels) of the median blur applied AFTER the BASEX transform
                       it is not normally used, but it can be a good way to get rid of high-frequency
                       artifacts in the transformed image. For example, it can reduce centerline noise.
         verbose - Set to True to see more output for debugging
         calc_speeds - determines if the speed distribution should be calculated
        """
        
        # make sure that the data is the right shape (1D must be converted to 2D)
        data = np.atleast_2d(data) # if passed a 1D array convert it to 2D
        if data.shape[0] == 1:
            self.ndim = 1
        elif data.shape[1] == 1:
            raise ValueError('Wrong input shape for data {0}, should be  (N1, N2) or (1, N), not (N, 1)'.format(data.shape))
        else:
            self.ndim = 2

        
        image = center_image(data, center=center, n=self.n, ndim=self.ndim)

        if symmetrize:
            #image = apply_symmetry(image)
            raise NotImplementedError

        if median_size>0:
            image = median_filter(image, size=median_size)

        if gaussian_blur>0:
            image = gaussian_filter(image, sigma=gaussian_blur)

        #Do the actual transform
        res = self._basex_transform(image)

        if self.calc_speeds:
            recon, speeds = res
        else:
            recon = res

        if post_median > 0:
            recon = median_filter(recon, size=post_median)


        if self.ndim == 1:
            recon = recon[0, :] # taking one row, since they are all the same anyway

        if self.scaling_correction:
            recon *= self._get_scaling_factor()

        if self.calc_speeds:
            return recon, speeds
        else:
            return recon



MAX_OFFSET = 4000


def generate_basis_sets(n=1001, nbf=500, verbose=True):
    """ 
    Generate the basis set for the BASEX method. 

    This function was adapted from the a matlab script provided by
    the Reisler group: BASIS2.m, with some optimizations.
    
    Typically, the number of basis functions will be (n-1)/2
    so that each pixel in the image is represented by its own basis function.

    Parameters:
    -----------
      n : integer : size of the basis set (pixels)
      nbf: integer: number of basis functions ?

    Returns:
    --------
      M, Mc : np.matrix
    """
    if n % 2 == 0:
        raise ValueError('The n parameter must be odd (more or less sure about it).')

    if n//2 < nbf:
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

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf)*32 - MAX_OFFSET, MAX_OFFSET) 
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


def get_left_right_matrices(M, Mc):
    left = inv(Mc.T.dot(Mc)).dot(Mc.T) 
    q=1;
    NBF=np.shape(M)[1] # number of basis functions
    E = np.identity(NBF)*q  # Creating diagonal matrix for regularization. (?)
    right = M.dot(inv((M.T.dot(M) + E)))
    return left, right
