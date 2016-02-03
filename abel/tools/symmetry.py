# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize 

def get_image_quadrants(IM, reorient=False, vertical_symmetry=False,
                        horizontal_symmetry=False, 
                        use_quadrants=(True, True, True, True)):
    """
    Given an image (m,n) return its 4 quadrants Q0, Q1, Q2, Q3
    as defined in abel.hansenlaw.iabel_hansenlaw

    Parameters
    ----------
    IM : 2D np.array
      Image data shape (rows, cols)

    reorient : boolean
      Reorient quadrants to match the orientation of Q0 (top-right)

    vertical_symmetry : boolean
      Exploit image symmetry to combine quadrants. Co-add Q0+Q1, and Q2+Q3

    horizontal_symmerty : boolean
      Exploit image symmetry to combine quadrants. Co-add Q1+Q2, and Q0+Q3

    use_quadrants : boolean tuple
      Include quadrant (Q0, Q1, Q2, Q3) in the symmetry combination(s)

    Returns
    -------
    Q0, Q1, Q2, Q3 : tuple of 2D np.arrays
      shape (rows//2+rows%2, cols//2+cols%2)
      reoriented to Q0 if True

    """
    IM = np.atleast_2d(IM)

    n, m = IM.shape

    n_c = n // 2  + n % 2
    m_c = m // 2  + m % 2

    # define 4 quadrants of the image
    # see definition in abel.hansenlaw.iabel_hansenlaw
    Q0 = IM[:n_c, -m_c:]
    Q1 = IM[:n_c, :m_c]
    Q2 = IM[-n_c:, :m_c]
    Q3 = IM[-n_c:, -m_c:]

    if vertical_symmetry:   # co-add quadrants
        Q0 = Q1 = Q0*use_quadrants[0]+Q1*use_quadrants[1]
        Q2 = Q3 = Q2*use_quadrants[2]+Q3*use_quadrants[3]

    if horizontal_symmetry:
        Q1 = Q2 = Q1*use_quadrants[1]+Q2*use_quadrants[2]
        Q0 = Q3 = Q0*use_quadrants[0]+Q3*use_quadrants[3]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))

    return Q0, Q1, Q2, Q3


def put_image_quadrants (Q, odd_size=True, vertical_symmetry=False, 
                         horizontal_symmetry=False):
    """
    Reassemble image from 4 quadrants Q = (Q0, Q1, Q2, Q3)
    The reverse process to get_image_quadrants()
    Qi defined in abel/hansenlaw.py
    
    Parameters
    ----------
    Q: tuple of np.array  (Q0, Q1, Q2, Q3)
       Image quadrants all oriented as Q0
       shape (rows//2+rows%2, cols//2+cols%2)

    even_size: boolean 
       Whether final image is even or odd pixel size
       odd size will trim 1 row from Q1, Q0, and 1 column from Q1, Q2
    
    vertical_symmetry : boolean
       Image symmetric about the vertical axis => Q0 == Q1, Q3 == Q2

    horizontal_symmerty : boolean
       Image symmetric about horizontal axis => Q2 == Q1, Q3 == Q0


    Returns  
    -------
    IM : np.array
        Reassembled image of shape (rows, cols) 
    """

    if vertical_symmetry:
        Q0 = Q1
        Q3 = Q2 

    if horizontal_symmetry:
        Q2 = Q1
        Q3 = Q0 

    if not odd_size:
        Top    = np.concatenate((np.fliplr(Q[1]), Q[0]), axis=1)
        Bottom = np.flipud(np.concatenate((np.fliplr(Q[2]), Q[3]), axis=1))
    else:
        # odd size image remove extra row/column added in get_image_quadrant()
        Top    = np.concatenate((np.fliplr(Q[1][:-1,:-1]), Q[0][:-1,:]), axis=1)
        Bottom = np.flipud(np.concatenate((np.fliplr(Q[2][:,:-1]), Q[3]), axis=1))

    IM = np.concatenate((Top,Bottom), axis=0)

    return IM


def center_image(data, center, n, ndim=2):
    """ This centers the image at the given center and makes it of size n by n"""
    
    Nh,Nw = data.shape
    n_2 = n//2
    if ndim == 1:
        cx = int(center)
        im = np.zeros((1, 2*n))
        im[0, n-cx:n-cx+Nw] = data
        im = im[:, n_2:n+n_2]
        # This is really not efficient
        # Processing 2D image with identical rows while we just want a
        # 1D slice 
        im = np.repeat(im, n, axis=0)

    elif ndim == 2:
        cx, cy = np.asarray(center, dtype='int')
        
        # Make an array of zeros that is large enough for cropping or padding:
        sz = 2*np.round(n + np.max((Nw, Nh)))
        im = np.zeros((sz, sz))
        
        # Set center of "zeros image" to be the data
        im[sz//2-cy:sz//2-cy+Nh, sz//2-cx:sz//2-cx+Nw] = data
        
        # Crop padded image to size n 
        # note the n%2 which return the appropriate image size for both 
        # odd and even images
        im = im[sz//2-n_2:n_2+sz//2+n%2, sz//2-n_2:n_2+sz//2+n%2]
        
    else:
        raise ValueError
    
    return im


def center_image_asym(data, center_column, n_vert, n_horz, verbose=False):
    """ This centers a (rectangular) image at the given center_column and makes it of size n_vert by n_horz"""

    if data.ndim > 2:
        raise ValueError("Array to be centered must be 1- or 2-dimensional")

    c_im = np.copy(data) # make a copy of the original data for manipulation
    data_vert, data_horz = c_im.shape
    pad_mode = str("constant")

    if data_horz % 2 == 0:
        # Add column of zeros to the extreme right to give data array odd columns
        c_im = np.pad(c_im, ((0,0),(0,1)), pad_mode, constant_values=0)
        data_vert, data_horz = c_im.shape # update data dimensions

    delta_h = int(center_column - data_horz//2)
    if delta_h != 0:
        if delta_h < 0: 
            # Specified center is to the left of nominal center
            # Add compensating zeroes on the left edge
            c_im = np.pad(c_im, ((0,0),(2*np.abs(delta_h),0)), pad_mode, constant_values=0)
            data_vert, data_horz = c_im.shape
        else:
            # Specified center is to the right of nominal center
            # Add compensating zeros on the right edge
            c_im = np.pad(c_im, ((0,0),(0,2*delta_h)), pad_mode, constant_values=0)
            data_vert, data_horz = c_im.shape

    if n_vert >= data_vert and n_horz >= data_horz:
        pad_up = (n_vert - data_vert)//2
        pad_down = n_vert - data_vert - pad_up
        pad_left = (n_horz - data_horz)//2
        pad_right = n_horz - data_horz - pad_left

        c_im = np.pad(c_im, ((pad_up,pad_down), (pad_left,pad_right)), pad_mode, constant_values=0)

    elif n_vert >= data_vert and n_horz < data_horz:
        pad_up = (n_vert - data_vert)//2
        pad_down = n_vert - data_vert - pad_up
        crop_left = (data_horz - n_horz)//2
        crop_right = data_horz - n_horz - crop_left
        if verbose:
            print("Warning: cropping %d pixels from the sides of the image" %crop_left)
        c_im = np.pad(c_im[:,crop_left:-crop_right], ((pad_up, pad_down), (0,0)), pad_mode, constant_values=0)

    elif n_vert < data_vert and n_horz >= data_horz:
        crop_up = (data_vert - n_vert)//2
        crop_down = data_vert - n_vert - crop_up
        pad_left = (n_horz - data_horz)//2
        pad_right = n_horz - data_horz - pad_left
        if verbose:
            print("Warning: cropping %d pixels from top and bottom of the image" %crop_up)
        c_im = np.pad(c_im[crop_up:-crop_down], ((0,0), (pad_left, pad_right)), pad_mode, constant_values=0)

    elif n_vert < data_vert and n_horz < data_horz:
        crop_up = (data_vert - n_vert)//2
        crop_down = data_vert - n_vert - crop_up
        crop_left = (data_horz - n_horz)//2
        crop_right = data_horz - n_horz - crop_left
        if verbose:
            print("Warning: cropping %d pixels from top and bottom and %d pixels from the sides of the image " %(crop_up, crop_left))
        c_im = c_im[crop_up:-crop_down,crop_left:-crop_right]

    else:
        raise ValueError('Input data dimensions incompatible with chosen basis set.')

    return c_im
