# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time


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


def onion_peeling_transform(IM, sym_lr=False, sym_ud=False, direction=None):
    # Abel-inversion algorithm from: Rev. Sci. Instrum. 67, 2257 (1996).
    # info about this implementation:
    # Rallis et al., Rev. Sci. Instrum. 85, 113105 (2014)

    # works only on the left side of an image.
    # i.e., for IM[i,j], the radial coordinate (r) increases with increasing j

    print("Warning: abel.onion_peeling_transform() is in early testing and \
           may not produce reasonable results")

    # Other methods use a Q0 oriented image, flip for onion use
    IM = IM[:, ::-1]  
    h, w = np.shape(IM)

    if w % 2 == 1:
        raise ValueError('Image width must be even')

    #if np.any(IM < 0):
    #    print('Image cannot have negative values, \
    #           setting negative values to zero')
    #    IM[IM < 0] = 0  # cannot have negative values

    # calculate val1 and val2, which are 2D arrays
    # of what appear to be scaling factors
    val1, val2 = init_abel(w, h)

    abel_arr = IM*0
    # initialize 2D array for final transform
    rest_arr = IM
    # initialize 2D array that will be manipulated during the transform
    vect = np.zeros(h)
    # initialize a 1D array that is temorarily used to store scaling factors

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
            #rest_arr[:, ic][rest_arr[:, ic] < 0] = 0

        for row_index in range(0, h):  # for row_index =1:ymax
            jdist = h - row_index
            vect[row_index] = val2[idist, jdist]

        abel_arr[:, col_index] = normfac * rest_col * vect.transpose()

    return abel_arr[:, ::-1] # flip back
