# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot

#################################################################################
#
#  Dasch onion-peeling deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1148 sect. D.
#    see PR #155
#
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2016-03-24 Steve Gibson - Python code framework
# 2015-12-29 Dhrubajyoti Das - three_point code and highlighting the Dasch paper
#                              see issue #61
#
#################################################################################


def onion_dasch_transform(IM, dr=1, direction="inverse"):
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
    inv_IM: 1D or 2D numpy array
        the inverse Abel transformed half-image 

    """

    if direction != 'inverse':
        raise ValueError('Forward "onion_dasch" transform not implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    # basis weight matrix 
    W = np.zeros_like(IM)

    # diagonal elements i = j, Eq. (11)
    I, J = np.diag_indices_from(IM) 
    W[I, J] = np.sqrt((2*J+1)**2 - 4*I**2)

    # upper triangle j > i
    Iu, Ju = np.triu_indices(IM.shape[0], k=1) 
    W[Iu, Ju] = np.sqrt((2*Ju+1)**2 - 4*Iu**2) -\
                np.sqrt((2*Ju-1)**2 - 4*Iu**2) 

    # operator used in Eq. (1)
    D = inv(W)   

    # one-line Abel transform - dot product of each row of IM with D
    inv_IM = np.tensordot(IM, D, axes=(1,1)) 

    if inv_IM.shape[0] == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr
