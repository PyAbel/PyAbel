# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot

def dasch_onion_peeling_transform(IM, dr=1, direction="inverse"):
    """ Onion-peeling deconvolution of Dasch  
        Applied Optics 31, 1146 (1992), page 1148 sect. D.

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

    if direction != 'inverse':
        raise ValueError('Forward "dasch_onion_peeling" transform not'
                         ' implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    cols, rows = IM.shape

    # transformed image
    AIM = np.zeros_like(IM)

    # weight matrix 
    W = np.zeros_like(IM)
    
    I, J = np.diag_indices_from(IM)
    Iu, Ju = np.triu_indices(IM.shape[0], k=1)
    
    for i in I:  
        W[i, i] = np.sqrt((2*i+1)**2 - 4*i**2)    # Eq. (11) j = i
    
    for i,j in zip(Iu, Ju):
        W[i, j] = np.sqrt((2*j+1)**2 - 4*i**2) -\
                  np.sqrt((2*j-1)**2 - 4*i**2)    # Eq. (11) j > i

    # operator used in Eq. (1)
    D = inv(W)   

    for i, P in enumerate(IM):
        AIM[i] = dot(W, P)   # Eq. (1)
    
    if AIM.shape[0] == 1:
        # flatten array
        AIM = AIM[0]

    return AIM*dr/cols   # normalization x(dr/cols)
