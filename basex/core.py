from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time
import os.path

import numpy as np
from numpy.linalg import inv
from scipy.ndimage import median_filter, gaussian_filter, map_coordinates

from .basis import generate_basis_sets
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
                    use_basis_set=None, verbose=True, calc_speeds=False):
        """ Initalize the BASEX class, preloading or generating the basis set.

        Parameters:
        -----------
          - N : odd integer: Abel inverse transform will be performed on a `n x n`
            area of the image
          - nbf: integer: number of basis functions ?
          - basis_dir : path to the directory for saving / loading the basis set coefficients.
          - use_basis_set: use the basis set stored as a text files, if
                  it provided, the following parameters will be ignored N, nbf, basis_dir
                  The expected format is a string of the form "some_basis_set_{}_1.bsc" where 
                  "{}" will be replaced by "" for the first file and "pr" for the second.
                  Gzip compressed text files are accepted.
          - verbose: Set to True to see more output for debugging
          - calc_speeds: determines if the speed distribution should be calculated

        """
        n = 2*(n//2) + 1 # make sure n is odd

        self.verbose = verbose
        self.calc_speeds = calc_speeds

        self.n = n
        self.nbf = nbf

        if self.verbose:
            t1 = time()

        basis_name = "basex_basis_{}_{}.npy".format(n, nbf)
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        
        if use_basis_set is not None:
            # load the matlab generated basis set
            M, Mc = parse_matlab(use_basis_set)
            left, right = get_left_right_matrices(M, Mc)

            self.n, self.nbf = M.shape # overwrite the provided parameters

        elif os.path.exists(path_to_basis_file):
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
                      'This may take a few minutes.',
                      'But don\'t worry, it will be saved to disk for future use.\n')

            M, Mc = generate_basis_sets(n, nbf, verbose=verbose)
            left, right = get_left_right_matrices(M, Mc)

            np.save(path_to_basis_file, (left, right, M, Mc))
            print('Basis set saved for later use to,')
            print(' '*10 + '{}'.format(path_to_basis_file))

        self.left, self.right, self.M, self.Mc = left, right, M, Mc



        if self.verbose:
            print('{:.2f} seconds'.format((time()-t1)))


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
            print('%.2f seconds' % (time()-t1))

        if self.calc_speeds:
            speeds = self.calculate_speeds(IM)
            return IM, speeds
        else:
            return IM


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
            image = median_filter(image,size=median_size)

        if gaussian_blur>0:
            image = gaussian_filter(image,sigma=gaussian_blur)

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

        if self.calc_speeds:
            return recon, speeds
        else:
            return recon


    def calculate_speeds(self, IM):
        # This section is to get the speed distribution.
        # The original matlab version used an analytical formula to get the speed distribution directly
        # from the basis coefficients. But, the C version of BASEX uses a numerical method similar to
        # the one implemented here. The difference between the two methods is negligable.
        """ Generating the speed distribution """

        if self.verbose:
            print('Generating speed distribution...')
            t1 = time()

        nx,ny = np.shape(IM)
        xi = np.linspace(-100, 100, nx)
        yi = np.linspace(-100, 100, ny)
        X,Y = np.meshgrid(xi,yi)

        polarIM, ri, thetai = reproject_image_into_polar(IM)

        speeds = np.sum(polarIM, axis=1)
        speeds = speeds[:self.n//2] #Clip off the corners

        if self.verbose:
            print('%.2f seconds' % (time()-t1))
        return speeds


def center_image(data, center, n, ndim=2):
    """ This centers the image at the given center and makes it of size n by n"""
    
    Nh,Nw = data.shape
    n_2 = n//2
    if ndim == 1:
        cx = int(center)
        im = np.zeros((1,2*n))
        im[0, n-cx:n-cx+Nw] = data
        im = im[:, n_2:n+n_2]
        # This is really not efficient
        # Processing 2D image with identical rows while we just want a
        # 1D slice 
        im = np.repeat(im, n, axis=0)

    elif ndim == 2:
        cx, cy = np.asarray(center, dtype='int')
        
        #make an array of zeros that is large enough for cropping or padding:
        sz = 2*np.round(n+np.max((Nw,Nh)))
        im = np.zeros((sz,sz))
        im[sz//2-cy:sz//2-cy+Nh, sz//2-cx:sz//2-cx+Nw] = data
        im = im[ sz//2-n_2-1:n_2+sz//2, sz//2-n_2-1:n_2+sz//2] #not sure if this exactly preserves the center
        print(np.shape(im))
    else:
        raise ValueError

    return im


def get_left_right_matrices(M, Mc):
    left = inv(Mc.T.dot(Mc)).dot(Mc.T) 
    q=1;
    NBF=np.shape(M)[1] # number of basis functions
    E = np.identity(NBF)*q  # Creating diagonal matrix for regularization. (?)
    right = M.dot(inv((M.T.dot(M) + E)))
    return left, right



# I got these next two functions from a stackoverflow page and slightly modified them.
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to get the speed distribution.
# If you figure it out, pease let me know! (danhickstein@gmail.com)
def reproject_image_into_polar(data, origin=None):
    """Reprojects a 2D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    nr = r.max()
    nt = ny//2

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr)
    theta_i = np.linspace(theta.min(), theta.max(), nt)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    X += origin[0] # We need to shift the origin
    Y += origin[1] # back to the lower-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((xi,yi)) # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr,nt))
    return output, r_i, theta_i


def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y

def cart2polar(x, y):
    """
    Transform carthesian coordinates to polar
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y,x)
    return r, theta

def polar2cart(r, theta):
    """
    Transform polar coordinates to carthesian
    """
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y
