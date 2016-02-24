# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from itertools import product


def three_point_transform(IM, basis_dir='.', direction="inverse", 
                          verbose=False):
    """
    Inverse Abel transformation using the algorithm of:
    Dasch, Applied Optics, Vol. 31, No. 8, 1146-1152 (1992).

    Parameters
    ----------
    IM : numpy array
        Raw data - a rows x cols array

    Returns
    -------
    inv_IM : numpy array
        Abel inversion of IM - a rows x cols array
    """
    IM = np.atleast_2d(IM)
    row, col = IM.shape
    D = get_bs_three_point_cached(col, basis_dir, verbose)
    inv_IM = np.zeros_like(IM)

    for i, P in enumerate(IM):
        inv_IM[i] = np.dot(D, P)
    return inv_IM


def OP_D(i, j):
    """
    Calculates three-point abel inversion operator D_ij,
    following Eq (6) in Dasch 1992 (Applied Optics).
    The original reference contains several typos.
    One correction is done in function OP1 following Karl Martin's PhD thesis
    See here:
    https://www.lib.utexas.edu/etd/d/2002/martinkm07836/martinkm07836.pdf
    """
    if j < i-1:
        D = 0.0
    elif j == i-1:
        D = OP0(i, j+1) - OP1(i, j+1)
    elif j == i:
        D = OP0(i, j+1) - OP1(i, j+1) + 2*OP1(i, j)
    elif i == 0 and j == 1:
        D = OP0(i, j+1) - OP1(i, j+1) + 2*OP1(i, j) - 2*OP1(i, j-1)
    elif j >= i+1:
        D = OP0(i, j+1) - OP1(i, j+1) + 2*OP1(i, j) - OP0(i, j-1) - OP1(i, j-1)
    else:
        raise(ValueError)
    return D


def OP0(i, j):
    """
    This is the I_ij(0) function in Dasch 1992, pg 1147, Eq (7)
    """
    if j < i or (j == i and i == 0):
        I0 = 0
    elif j == i and i != 0:
        I0 = np.log((((2*j+1)**2 - 4*i**2)**0.5 + 2*j+1)/(2*j))/(2*np.pi)
    elif j > i:
        I0 = np.log((((2*j+1)**2 - 4*i**2)**0.5 + 2*j+1) /
                    (((2*j-1)**2 - 4*i**2)**0.5 + 2*j-1))/(2*np.pi)
    else:
        raise(ValueError)
    return I0


def OP1(i, j):
    """
    This is the I_ij(1) function in Dasch 1992, pg 1147, Eq (7)
    """
    if j < i:
        I1 = 0
    elif j == i:
        I1 = ((2*j+1)**2 - 4*i**2)**0.5/(2*np.pi) - 2*j*OP0(i, j)
    elif j > i:
        I1 = (((2*j+1)**2 - 4*i**2)**0.5 -
              ((2*j-1)**2 - 4*i**2)**0.5)/(2*np.pi) - 2*j*OP0(i, j)
    else:
        raise(ValueError)
    return I1


def get_bs_three_point_cached(col, basis_dir='.', verbose=False):
    """
    Internal function.

    Gets thre_point operator matrix corresponding to specified image size,
    using the disk as a cache.
    (i.e., load from disk if they exist, if not, calculate them
    and save a copy on disk)

    Parameters
    ----------
    col : integer
        Width of image to be inverted using three_point method.
        Three_point operator matrix will be of size (col x col)
    basis_dir : string
        path to the directory for saving / loading
        the three_point operator matrix.
        If None, the operator matrix will not be saved to disk.
    verbose : True/False
        Set to True to see more output for debugging
    """
    basis_name = "three_point_basis_{}_{}.npy".format(col, col)
    D = None
    if basis_dir is not None:
        path_to_basis_file = os.path.join(basis_dir, basis_name)
        if os.path.exists(path_to_basis_file):
            if verbose:
                print("Loading three_point operator matrix...")
            try:
                D = np.load(path_to_basis_file)
            except ValueError:
                raise
            except:
                raise
    if D is None:
        if verbose:
            print("A suitable operator matrix was not found.",
                  "A new operator matrix will be generated.",
                  "This may take a few minutes.", end=" ")
            if basis_dir is not None:
                print("But don\'t worry, it will be saved to disk \
                    for future use.\n")
            else:
                pass
        D = np.zeros((col, col))

        for i, j in product(range(col), range(col)):
            D[i, j] = OP_D(i, j)
        if basis_dir is not None:
            np.save(path_to_basis_file, D)
            if verbose:
                print("Operator matrix saved for later use to,")
                print(' '*10 + '{}'.format(path_to_basis_file))
    return D


def iabel_three_point(data, center,
                      dr=1.0, basis_dir='./', verbose=False,
                      direction='inverse'):
    """
    This function splits the image into two halves,
    sends each half to iabel_three_point_transform(),
    stitches the output back together,
    and returns the full transform to the user.

    Parameters
    -----------
    data : NxM numpy array
        The raw data is presumed to be symmetric
        about the vertical axis.
    center : integer or tuple (x,y)
        The location of the symmetry axis
        (center column index of the image) or
        the center of the image in (x,y) format.
    dr : float
        Size of one pixel in the radial direction
    basis_dir : string
        path to the directory for saving / loading the three_point
        operator matrix.
        If None, the operator matrix will not be saved to disk.
    verbose : True/False
        Set to True to see more output for debugging
    direction : str
        The type of Abel transform to be performed.
        Currently only accepts value 'inverse'

    Returns
    -------
    inv_IM : numpy array
        Abel inversion of IM - a rows x cols array
    """

    if direction != 'inverse':
        raise ValueError('Forward three_point transform not implemented')

    # sanity checks for center
    # 1. If center is tuple, only take the second value inside it
    if isinstance(center, int):
        pass
    elif isinstance(center, tuple):
        _, center = center  # extracting y from (x,y)
    else:
        raise ValueError('Center must be an integer or tuple.')

    # 2. center index must be >= 0 (possibly >= 2 for the transform)
    # 3. center index must be < # of columns in raw data
    data = np.atleast_2d(data)
    row, col = data.shape
    if not 0 <= center <= col-1:
        raise ValueError('Center column index invalid.')

    # cut data in half
    # each half has the center column at one edge
    left_half, right_half = data[:, 0:center+1], data[:, center:]

    # mirror left half
    left_half = np.fliplr(left_half)

    # transform both halves
    inv_left = iabel_three_point_transform(left_half, basis_dir, verbose)
    inv_right = iabel_three_point_transform(right_half, basis_dir, verbose)

    # undo mirroring of left half
    inv_left = np.fliplr(inv_left)

    # stitch both halves back together
    # (extra) center column is excluded from left half
    inv_IM = np.hstack((inv_left[:, :-1], inv_right))

    # scale output by dr
    inv_IM = inv_IM/dr

    # if data is 1-dimensional, output ought to be 1-dimensional
    if row == 1:
        inv_IM = inv_IM[0]
    return inv_IM
