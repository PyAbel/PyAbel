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

##############################################################################
#
#  Dasch three-point deconvolution
#    as described in Applied Optics 31, 1146 (1992), page 1147 sect. B.
#    see PR #155
#
# 2016-03-28 Steve Gibson - Python code framework
# 2016-03-25 Dan Hickstein - one line Abel transform
# 2015-12-29 Dhrubajyoti Das - original three_point code and 
#                              highlighting the Dasch paper see issue #61
#                              https://github.com/PyAbel/PyAbel/issues/61
#
###############################################################################


def three_point_transform(IM, basis_dir='.', dr=1, direction="inverse"):
    """ Three point deconvolution of Dasch  
        Applied Optics 31, 1146 (1992), page 1148 sect. C.

    Parameters
    ----------
    IM : 1D or 2D numpy array
        right-side half-image (or quadrant)

    basis_dir: str
        path to the directory for saving / loading
        the three_point operator matrix.
        If None, the operator matrix will not be saved to disk.

    dr : float
        not used (grid size for other algorithms).

    direction: str
        only the `direction="inverse"` transform is available.


    Returns
    -------
    inv_IM: 1D or 2D numpy array
        the inverse Abel transformed half-image 

    """

    if direction != 'inverse':
        raise ValueError('Forward "three_point" transform not implemented')

    # make sure that the data has 2D shape
    IM = np.atleast_2d(IM)

    rows, cols = IM.shape
    
    if cols < 4: 
        raise ValueError('"three_point" requires image width (cols) > 3')

    D = abel.tools.basis.get_bs_cached("three_point", cols, basis_dir=basis_dir)

    inv_IM = abel.tools.basis.abel_transform(IM, D)

    if rows == 1:
        inv_IM = inv_IM[0]  # flatten array

    return inv_IM/dr

def _bs_three_point(cols):
    """basis function for three_point.
    
    Parameters
    ----------
    cols : int
        width of the image
    """

    # basis function Eq. (7)  for j >= i
    def I0diag(i, j):
        return np.log( (np.sqrt((2*j+1)**2-4*i**2) + 2*j+1)/(2*j) )/(2*np.pi)

    # j > i
    def I0(i, j):
        return np.log(((np.sqrt((2*j+1)**2 - 4*i**2) + 2*j+1))/ 
                       (np.sqrt((2*j-1)**2 - 4*i**2) + 2*j-1))/(2*np.pi) 

    # i = j  NB minus -2I_ij typo in Dasch paper
    def I1diag(i, j):
        return np.sqrt((2*j+1)**2 - 4*i**2)/(2*np.pi) - 2*j*I0diag(i, j)

    # j > i
    def I1(i, j):
        return (np.sqrt((2*j+1)**2 - 4*i**2) -
                np.sqrt((2*j-1)**2 - 4*i**2))/(2*np.pi) - 2*j*I0(i, j)

    D = np.zeros((cols, cols))

    # matrix indices ------------------
    # i = j
    I, J = np.diag_indices(cols)
    I = I[1:]
    J = J[1:]  # drop special cases (0,0), (0,1)

    # j = i - 1
    Ib, Jb = I, J-1

    # j = i + 1
    Iu, Ju = I-1, J
    Iu = Iu[1:]  # drop special case (0, 1)
    Ju = Ju[1:] 

    # j > i + 1
    Iut, Jut = np.triu_indices(cols, k= 2)
    Iut = Iut[1:]  # drop special case (0, 2)
    Jut = Jut[1:] 

    # D operator matrix ------------------
    # j = i - 1
    D[Ib, Jb] = I0diag(Ib, Jb+1) - I1diag(Ib, Jb+1)

    # j = i
    D[I, J] = I0(I, J+1) - I1(I, J+1) + 2*I1diag(I, J)

    # j = i + 1
    D[Iu, Ju] = I0(Iu, Ju+1) - I1(Iu, Ju+1) + 2*I1(Iu, Ju) -\
                I0diag(Iu, Ju-1) - I1diag(Iu, Ju-1)

    # j > i + 1
    D[Iut, Jut] = I0(Iut, Jut+1) - I1(Iut, Jut+1) + 2*I1(Iut, Jut) -\
                  I0(Iut, Jut-1) - I1(Iut, Jut-1)

    # special cases (that switch between I0, I1 cases)
    D[0, 2] = I0(0, 3) - I1(0, 3) + 2*I1(0, 2) - I0(0, 1) - I1(0, 1) 
    D[0, 1] = I0(0, 2) - I1(0, 2) + 2*I1(0, 1) - 1/np.pi
    D[0, 0] = I0(0, 1) - I1(0, 1) + 1/np.pi

    return D
