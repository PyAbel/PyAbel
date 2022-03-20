# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import shift

################################################################################
#
# Onion peeling algorithm, also known as "back projection"
#
# This algorithm was adapted by Dan Hickstein from the original Matlab 
# implementation, created by Chris Rallis and Eric Wells of 
# Augustana University, and described in this paper:
#
#  https://scitation.aip.org/content/aip/journal/rsi/85/11/10.1063/1.4899267
#
# The algorithm actually originates from this 1996 RSI paper by Bordas et al:
#
#  https://scitation.aip.org/content/aip/journal/rsi/67/6/10.1063/1.1147044
#
# 2016-03-19: Jacobian intensity correction, allow negative image values
#             Steve Gibson and Dan Hickstein
# 2015-12-16: Python version of the Matlab code, by Dan Hickstein see issue #56
#
################################################################################

def _init_abel_vec(xc, yc):
    # vectorized 
    i = np.arange(xc, dtype=int)
    j = np.arange(yc, dtype=int)

    Ip1, I = np.meshgrid(i+1, i)
    Ijp1, Iip1 = np.meshgrid(i+1, i+1)

    val1 = np.arcsin(np.triu(Iip1/Ijp1)) - np.arcsin(np.triu(I/Ip1))

    Jp1, I1 = np.meshgrid(j+1, i+1)
    val2 = 1.0/I1[:]

    return val1, val2

def _init_abel(xc, yc):
    # this seems like it could be vectorized pretty easily
    val1 = np.zeros((xc, xc))
    val2 = np.zeros((xc, yc))

    for ii in range(xc):
        for jj in range(ii, xc):
            val1[ii, jj] = np.arcsin((ii+1)/(jj+1)) - \
                           np.arcsin((ii)/(jj+1))

        val2[ii, :] = 1.0/(ii+1)

    return val1, val2


def onion_bordas_transform(IM, dr=1, direction="inverse", shift_grid=True,
                           **kwargs):
    r""":doc:`Onion peeling (or back projection)
    <transform_methods/onion_bordas>` inverse Abel transform.

    This algorithm was adapted by Dan Hickstein from the original Matlab 
    implementation, created by Chris Rallis and Eric Wells of 
    Augustana University, and described in

    C. E. Rallis, T. G. Burwitz, P. R. Andrews, M. Zohrabi, R. Averin, S. De,
    B. Bergues, B. Jochim, A. V. Voznyuk, N. Gregerson, B. Gaire,
    I. Znakovskaya, J. McKenna, K. D. Carnes, M. F. Kling, I. Ben-Itzhak,
    E. Wells,
    "Incorporating real time velocity map image reconstruction into closed-loop
    coherent control",
    `Rev. Sci. Instrum. 85, 113105 (2014)
    <https://doi.org/10.1063/1.4899267>`__.

    The algorithm actually originates from

    C. Bordas, F. Paulig,
    "Photoelectron imaging spectrometry: Principle and inversion method",
    `Rev. Sci. Instrum. 67, 2257–2268 (1996)
    <https://doi.org/10.1063/1.1147044>`__.

    This function operates on the "right side" of an image. i.e. it works on
    just half of a cylindrically symmetric image. Unlike the other transforms,
    the image origin should be at the left edge, not mid-pixel. This
    corresponds to an even-width full image.
                           
    However, ``shift_grid=True`` (default) provides the typical behavior,
    where the image origin corresponds to the pixel center in the 0th column.

    To perform a onion-peeling transorm on a whole image, use ::
    
        abel.Transform(image, method='onion_bordas').transform

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    dr : float
        sampling size (=1 for pixel images), used for Jacobian scaling.
        The resulting inverse transform is simply scaled by 1/`dr`.

    direction : str
        only the inverse transform is currently implemented.
   
    shift_grid : bool
        place the image origin on the grid (left edge) by shifting the image
        1/2 pixel to the left.

    Returns
    -------
    AIM: 1D or 2D numpy array
        the inverse Abel transformed half-image 

    """

    if direction != 'inverse':
        raise ValueError('Forward "onion_bordas" transform not implemented')

    # make sure that the data is the right shape (1D must be converted to 2D):
    IM = np.atleast_2d(IM.copy())
    
    # onion-peeling uses grid rather than pixel values, 
    # odd shaped whole images require shift image (-1/2, -1/2)
    
    if shift_grid:
        IM = shift(IM, (0,-0.5), mode='nearest')

    # we would like to work from the outside to the inside of the image, 
    # so flip the image to put the "outside" at low index values.
    IM = np.fliplr(IM)

    h, w = np.shape(IM)

    # calculate val1 and val2, which are 2D arrays
    # of what appear to be scaling factors
    val1, val2 = _init_abel(w, h) 

    abel_arr = np.zeros_like(IM)
    # initialize 2D array for final transform
    rest_arr = IM
    # initialize 2D array that will be manipulated during the transform
    vect = np.zeros(h)
    # initialize a 1D array that is temporarily used to store scaling factors

    for col_index in range(1, w):
        # iterate over the columns (x-values) in the slice space

        idist = w - col_index
        # this is basically the x-coordinate
        # or "distance in i direction", I guess
        rest_col = rest_arr[:, col_index-1]
        # a 1D column. This is the piece of the onion we are on.
        normfac = 1 / val1[idist, idist]

        for i in range(idist)[::-1]:
            rest_arr[:, w-i-1] -= rest_col * normfac * val1[i, idist]

        for row_index in range(h):  # for row_index =1:ymax
            vect[row_index] = val2[idist, h-row_index-1]

        abel_arr[:, col_index] = normfac * rest_col * vect.transpose()

    abel_arr = np.c_[abel_arr[:, 1:],abel_arr[:, -1]]
    abel_arr = np.fliplr(abel_arr) # flip back

    # shift back to pixel grid
    if shift_grid:
        abel_arr = shift(abel_arr, (0, 1.0), mode='nearest')
    
    if abel_arr.shape[0] == 1:
        # flatten array
        abel_arr = abel_arr[0]
        
    return abel_arr/(2*dr)  # x1/2 for 'correct' normalization   
