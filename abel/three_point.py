# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from itertools import product

def iabel_three_point_transform(IM):
    """ Inverse Abel transformation using the algorithm of: 
        Dasch, Applied Optics, Vol. 31, No. 8, 1146-1152 (1992).

        Parameters:
        ----------
         - IM: a rows x cols numpy array 

         Returns:
         --------
         - inv_IM: a rows x cols numpy array containing the Abel inversion of IM
    """
    IM = np.atleast_2d(IM)
    row, col = IM.shape
    D = np.zeros((col, col))

    for i,j in product(range(col), range(col)):
        D[i,j] = OP_D(i,j)

    inv_IM = np.zeros_like(IM)

    for i, P in enumerate(IM):
        F = np.zeros(col)
        for j,k in product(range(col), range(col)):
            F[j] += D[j,k] * P[k]

        inv_IM[i] = F

    return inv_IM

def OP_D(i,j):
    """
    Calculate three-point abel inversion operator Di,j
    The formula followed Dasch 1992 (Applied Optics) which contains several typos.
    One correction is done in function OP1 follow Martin's PhD thesis
    """
    
    if j < i-1:
        D = 0.0
    elif j == i-1:
        D = OP0(i,j+1) - OP1(i,j+1)
    elif j == i:
        D = OP0(i,j+1) - OP1(i,j+1) + 2*OP1(i,j)
    elif i == 0 and j == 1:
        D = OP0(i,j+1) - OP1(i,j+1) + 2*OP1(i,j) - 2*OP1(i,j-1)
    elif j >= i+1:
        D = OP0(i,j+1) - OP1(i,j+1) + 2*OP1(i,j) - OP0(i,j-1) - OP1(i,j-1)
    else:
        raise(ValueError)    
    
    return D

def OP0(i,j):
    
    if j < i or (j == i and i == 0):
        I0 = 0
    elif j == i and i != 0:
        I0 = np.log((((2*j+1)**2 - 4*i**2)**0.5 + 2*j+1)/(2*j))/(2*np.pi)
    elif j > i:
        I0 = np.log((((2*j+1)**2 - 4*i**2)**0.5 + 2*j+1)/(((2*j-1)**2 - 
                                                           4*i**2)**0.5 + 2*j-1))/(2*np.pi)
    else:
        raise(ValueError)
        
    return I0

def OP1(i,j):
    if j < i:
        I1 = 0
    elif j == i:
        I1 = ((2*j+1)**2 - 4*i**2)**0.5/(2*np.pi) - 2*j*OP0(i,j)
    elif j > i:
        I1 = (((2*j+1)**2 - 4*i**2)**0.5 - ((2*j-1)**2 - 4*i**2)**0.5)/(2*np.pi) - 2*j*OP0(i,j)
    else:
        raise(ValueError)
        
    return I1

def iabel_three_point(data, center, dr = 1.0):  
    """ This function splits the image into two halves, 
        sends each half to iabel_three_point_transform(), 
        stitches the output back together,
        and returns the full transform to the user.

    Parameters:
    -----------
        - data:   NxM numpy array
                  The raw data is presumed to be symmetric about the vertical axis. 
        - center: * integer - 
                    The location of the symmetry axis (center column index of the image). 
                  * tuple (x,y) - 
                    The center of the image in (x,y) format.
        - dr:     float - 
                  Size of one pixel in the radial direction
    """

    # sanity checks for center

    # cut data in half
    # each half has the center column at one edge
    left_half, right_half = data[:,0:center+1], data[:,center:]

    # mirror left half
    left_half = np.fliplr(left_half)

    # transform both halves
    inv_left = iabel_three_point_transform(left_half)
    inv_right = iabel_three_point_transform(right_half)

    # undo mirroring of left half
    inv_left = np.fliplr(inv_left)

    # stitch both halves back together
    # (extra) center column is excluded from left half
    inv_IM = np.hstack((inv_left[:,:-1], inv_right))

    return inv_IM