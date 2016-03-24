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
    """ Onion-peeling deconvolution of Dasch et al.
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

    # make sure that the data is the right shape 
    #(1D must be converted to 2D):
    IM = np.atleast_2d(IM)

    cols, rows = IM.shape
    AIM = np.zeros_like(IM)

    W = np.zeros_like(IM)
    
    I, J = np.diag_indices_from(IM)
    Iu, Ju = np.triu_indices(W.shape[0], k=1)

    for i in I:
        W[i, i] = np.sqrt((2*i+1)**2 - 4*i**2)
    
    for i,j in zip(Iu, Ju):
        W[i, j] = np.sqrt((2*j+1)**2 - 4*i**2) -\
                  np.sqrt((2*j-1)**2 - 4*i**2)

    D = inv(W)

    for i, P in enumerate(IM):
        AIM[i] = dot(W, P)
    
    if AIM.shape[0] == 1:
        # flatten array
        AIM = AIM[0]

    return AIM*dr/cols
