# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot


def two_point_transform(IM, dr=1, direction="inverse"):
    """ Two-point deconvolution of Dasch  
        Applied Optics 31, 1146 (1992), page 1148 sect. C.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    dr : float
        not used (grid size for other algorithms)

    direction: str
        only the `direction="inverse"` transform is currently implemented


    Returns
    -------
    AIM: 1D or 2D numpy array
        the inverse Abel transformed half-image 

    """

    # Eq. (9)  for j > i
    def J(i, j): 
       return np.log((np.sqrt((j+1)**2-i**2)+j+1)/\
                     (np.sqrt((j-1)**2-i**2)+j))/np.pi

    if direction != 'inverse':
        raise ValueError('Forward "two_point" transform not'
                         ' implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    cols, rows = IM.shape

    # transformed image
    AIM = np.zeros_like(IM)

    # operator
    D = np.zeros_like(IM)

    Iu, Ju = np.triu_indices(cols, k=0)  # upper triangle j >= i

    # Eq. (8, 9)
    for i, j in zip(Iu[1:], Ju[1:]):    # drop [0, 0]
        if j > i+1:
            D[i, j] = J(i, j) - J(i, j-1)
        elif j > i: 
            D[i, j] = J(i, j)

    D[0, 0] = 2/np.pi  # special case i=j=0

    for i, P in enumerate(IM):
        AIM[i] = dot(D, P)   # Eq. (1)

    if AIM.shape[0] == 1:
        # flatten array
        AIM = AIM[0]

    return AIM*dr
