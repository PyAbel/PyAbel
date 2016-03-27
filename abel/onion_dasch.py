# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot

################################################################################
#
#  Dasch onion-peeling deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1148 sect. D.
#    see PR #155
#
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2016-03-24 Steve Gibson - Python code framework
# 2015-12-29 Dhrubajyoti Das - three_point code and highlighting the Dasch paper
#            see issue #61,  https://github.com/PyAbel/PyAbel/issues/61
#
################################################################################


def onion_dasch_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    """ Onion-peeling deconvolution of Dasch  
        Applied Optics 31, 1146 (1992), page 1148 sect. D.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir : str
        path to the directory for saving / loading
        the onion_dasch operator matrix.
        If None, the operator matrix will not be saved to disk.

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

    rows, cols = IM.shape

    D = abel.tools.basis.get_bs_cached("onion_dasch", cols, basis_dir=basis_dir)

    inv_IM = _onion_dasch_core_transform(IM, D)

    if inv_IM.shape[0] == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr

def _onion_dasch_core_transform(IM, D):
    """Inverse Abel transform (onion peeling - Dasch version)
       using a given D-operator basis matrix.
    """
    # one-line Abel transform - dot product of each row of IM with D
    return np.tensordot(IM, D, axes=(1, 1))

def _bs_onion_dasch(cols):
    """basis function for onion_dasch.
    
    Parameters
    ----------
    cols : int
        width of the image
    """

    # basis weight matrix 
    W = np.zeros((cols, cols))

    # diagonal elements i = j, Eq. (11)
    I, J = np.diag_indices(cols) 
    W[I, J] = np.sqrt((2*J+1)**2 - 4*I**2)

    # upper triangle j > i,  Eq. (11)
    Iu, Ju = np.triu_indices(cols, k=1) 
    W[Iu, Ju] = np.sqrt((2*Ju+1)**2 - 4*Iu**2) -\
                np.sqrt((2*Ju-1)**2 - 4*Iu**2) 

    # operator used in Eq. (1)
    D = inv(W)   

    return D
