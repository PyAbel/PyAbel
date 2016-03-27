# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path
import numpy as np
import abel
from scipy.linalg import inv
from scipy import dot

################################################################################
#
#  Dasch two-point deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1148 sect. C.
#    see PR #155
#
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2016-03-24 Steve Gibson - Python code framework
# 2015-12-29 Dhrubajyoti Das - three_point code and highlighting the Dasch paper
#                              see issue #61
#
################################################################################


def two_point_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    """ Two-point deconvolution of Dasch  
        Applied Optics 31, 1146 (1992), page 1148 sect. C.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str
        path to the directory for saving / loading
        the two_point operator matrix.
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
        raise ValueError('Forward "two_point" transform not implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape

    D = abel.tools.basis.get_bs_cached("two_point", cols, basis_dir=basis_dir)

    inv_IM = _two_point_core_transform(IM, D)

    if rows == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr

def _two_point_core_transform(IM, D):
    """Inverse Abel transform (two point) 
       using a given D-operator basis matrix.
    """
    # one-line Abel transform - dot product of each row of IM with D
    return np.tensordot(IM, D, axes=(1, 1))

def _bs_two_point(cols):
    """basis function for two_point.
    
    Parameters
    ----------
    cols : int
        width of the image
    """

    # basis function Eq. (9)  for j >= i
    def J(i, j): 
       return np.log((np.sqrt((j+1)**2-i**2) + j+1)/\
                     (np.sqrt(j**2-i**2) + j))/np.pi

    # Eq. (8, 9) D-operator basis, is 0 for j < i
    D = np.zeros((cols, cols))

    # diagonal i == j
    Ii, Jj = np.diag_indices(cols) 
    Ii = Ii[1:]  # exclude special case i=j=0
    Jj = Jj[1:]
    D[Ii, Jj] = J(Ii, Jj)

    # upper triangle j > i
    Iu, Ju = np.triu_indices(cols, k=1)
    Iu = Iu[1:]  # exclude special case [0, 1]
    Ju = Ju[1:]
    D[Iu, Ju] = J(Iu, Ju) - J(Iu, Ju-1)

    # special cases
    D[0, 1] = J(0, 1) - 2/np.pi
    D[0, 0] = 2/np.pi

    return D
