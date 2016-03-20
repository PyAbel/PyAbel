# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage.interpolation import shift

################################################################################
#
# Onion peeling algorithm, also known as "back projection"
#
# This algorithm was adapted by Dan Hickstein from the original Matlab 
# implementation, created by Chris Rallis and Eric Wells of 
# Augustana University, and described in this paper:
#
#  http://scitation.aip.org/content/aip/journal/rsi/85/11/10.1063/1.4899267
#
# The algorithm actually originates from this 1996 RSI paper by Bordas et al:
#
#  http://scitation.aip.org/content/aip/journal/rsi/67/6/10.1063/1.1147044
#
# 2016-03-19: Jacobian intensity correction, allow negative image values
#             Steve Gibson and Dan Hickstein
# 2015-12-16: Python version of the Matlab code, by Dan Hickstein see issue #56
#
################################################################################

def init_abel(xc, yc):
    # this seems like it could be vectorized pretty easily
    val1 = np.zeros((xc+1, xc+1))
    val2 = np.zeros((xc+1, yc+1))

    for ii in range(0, xc+1):
        for jj in range(ii, xc+1):
            val1[ii, jj] = np.arcsin((ii+1)/float(jj+1)) - \
                np.arcsin((ii)/float(jj+1))

    for idist in range(0, xc+1):
        for jdist in range(0, yc+1):
            val2[idist, jdist] = np.sqrt((idist+1)**2+jdist**2)/(idist+1)

    return val1, val2


def onion_peeling_transform(IM, dr=1, direction="inverse", shift_grid=False):
    r"""Onion peeling (or back projection) inverse Abel transform.

    This function operates on the "right side" of an image. i.e.
    it works on just half of a cylindrically symmetric image.
    Unlike the other transforms, the left edge should be the
    image center, not mid-first pixel. This corresponds to an
    even-width full image. If not, set `shift_grid=True`. 

    To perform a onion-peeling transorm on a whole image, use ::
    
        abel.Transform(image, method='onion_peeling').transform

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    dr : float
        not used (grid size for other algorithms)

    direction: str
        only the `direction="inverse"` transform is currently implemented
   
    shift_grid: boolean
        place width-center on grid (bottom left pixel) by shifting image 
        center (-1/2, -1/2) pixel 

    Returns
    -------
    AIM: 1D or 2D numpy array
        the inverse Abel transformed half-image 

    """


    # onion-peeling uses grid rather than pixel values, 
    # odd shaped whole images require shift image (-1/2, -1/2)
    if shift_grid:
        IM = shift(IM, -1/2)

    IM = np.atleast_2d(IM)

    # The original pythod code operated on the left-half image
    # Other methods use a right-half oriented image, flip for common use
    IM = IM[:, ::-1]  

    h, w = np.shape(IM)

    # calculate val1 and val2, which are 2D arrays
    # of what appear to be scaling factors
    val1, val2 = init_abel(w, h)

    abel_arr = IM*0
    # initialize 2D array for final transform
    rest_arr = IM
    # initialize 2D array that will be manipulated during the transform
    vect = np.zeros(h)
    # initialize a 1D array that is temporarily used to store scaling factors

    for col_index in range(0, w):
        # iterate over the columns (x-values) in the slice space
        # if col_index%100 == 0: print col_index

        idist = w - col_index
        # this is basically the x-coordinate
        # or "distance in i direction", I guess
        rest_col = rest_arr[:, col_index]
        # a 1D column. This is the piece of the onion we are on.
        normfac = 1 / val1[idist, idist]

        for i in range(0, idist)[::-1]:
            ic = w - i - 1
            rest_arr[:, ic] = rest_arr[:, ic] - \
                rest_col * normfac * val1[i, idist]

        for row_index in range(0, h):  # for row_index =1:ymax
            jdist = h - row_index
            vect[row_index] = val2[idist, jdist]

        abel_arr[:, col_index] = normfac * rest_col * vect.transpose()

    abel_arr = abel_arr[:, ::-1] # flip back

    # Jacobian intensity correction `x 1/r` @DanHickstein #53
    # this factor may be better incorporated in the code above
    if abel_arr.shape[0] > 1:
        x = np.linspace(2,w,w)
        y = np.linspace(2,h,h)[::-1]

        X,Y = np.meshgrid(x,y)

        R = np.sqrt(X**2 + Y**2)
        abel_arr /= R
    else:
        abel_arr = abel_arr[0]
        x = np.linspace(1,w,w)
        abel_arr /= x

    # shift back to pixel grid
    if shift_grid:
        abel_arr = shift(abel_arr, 1/2)

    return abel_arr   
