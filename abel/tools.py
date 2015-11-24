# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates

def calculate_speeds(IM):
    """ This performs an angular integration of the image and returns the one-dimentional intensity profile 
        as a function of the radial coordinate. It assumes that the image is properly centered. 
        
     Parameters
     ----------
      - IM: a NxN ndarray.
      
     Returns
     -------
      - speeds: a 1D array of the integrated intensity versus the radial coordinate.
     """
    
    polarIM, ri, thetai = reproject_image_into_polar(IM)

    speeds = np.sum(polarIM, axis=1)
    
    # Clip the data corresponding to the corners, since these pixels contain incomplete information
    n = np.min(np.shape(IM))//2     # find the shortest radial coordinate
    speeds = speeds[:n//2]          # clip the 1D data

    return speeds


def get_image_quadrants(img, reorientate=False):
    """
    Given an image (m,n) reuturn its 4 quadrants Q0, Q1, Q2, Q3
    as defined in abel.hansenlaw.iabel_hansenlaw

    Parameters:
      - img: 1D or 2D array
      - reorientate: reorientate image as required by abel.hansenlaw.iabel_hansenlaw
    """
    img = np.atleast_2d(img)

    n, m = img.shape

    n_c = n//2 + n%2
    m_c = m//2 + m % 2

    # define 4 quadrants of the image
    # see definition in abel.hansenlaw.iabel_hansenlaw
    Q1 = img[:n_c, :m_c]
    Q2 = img[-n_c:, :m_c]
    Q0 = img[:n_c, -m_c:]
    Q3 = img[-n_c:, -m_c:]

    if reorientate:
        Q0 = np.fliplr(Q0)
        Q2 = np.flipud(Q2)
        Q3 = np.fliplr(np.flipud(Q3))

    return Q0, Q1, Q2, Q3


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

def center_image_asymmetric(data, center_column, n_vert, n_horz, verbose=False):
    """ This centers a (rectangular) image at the given center_column and makes it of size n_vert by n_horz"""

    if data.ndim > 2:
        raise ValueError("Array to be centered must be 1- or 2-dimensional")

    c_im = np.copy(data) # make a copy of the original data for manipulation
    data_vert, data_horz = c_im.shape

    if data_horz % 2 == 0:
        # Add column of zeros to the extreme right to give data array odd columns
        c_im = np.lib.pad(c_im, ((0,0),(0,1)), 'constant', constant_values=0)
        data_vert, data_horz = c_im.shape # update data dimensions

    delta_h = int(center_column - data_horz//2)
    if delta_h != 0:
        if delta_h < 0: 
            # Specified center is to the left of nominal center
            # Add compensating zeroes on the left edge
            c_im = np.lib.pad(c_im, ((0,0),(2*np.abs(delta_h),0)), 'constant', constant_values=0)
            data_vert, data_horz = c_im.shape
        else:
            # Specified center is to the right of nominal center
            # Add compensating zeros on the right edge
            c_im = np.lib.pad(c_im, ((0,0),(0,2*delta_h)), 'constant', constant_values=0)
            data_vert, data_horz = c_im.shape

    if n_vert >= data_vert and n_horz >= data_horz:
        pad_up = (n_vert - data_vert)//2
        pad_down = n_vert - data_vert - pad_up
        pad_left = (n_horz - data_horz)//2
        pad_right = n_horz - data_horz - pad_left
        c_im = np.lib.pad(c_im, ((pad_up,pad_down), (pad_left,pad_right)), 'constant', constant_values=0)

    elif n_vert >= data_vert and n_horz < data_horz:
        pad_up = (n_vert - data_vert)//2
        pad_down = n_vert - data_vert - pad_up
        crop_left = (data_horz - n_horz)//2
        crop_right = data_horz - n_horz - crop_left
        if verbose:
            print("Warning: cropping %d pixels from the sides of the image" %crop_left)
        c_im = np.lib.pad(c_im[:,crop_left:-crop_right], ((pad_up, pad_down), (0,0)), 'constant', constant_values=0)

    elif n_vert < data_vert and n_horz >= data_horz:
        crop_up = (data_vert - n_vert)//2
        crop_down = data_vert - n_vert - crop_up
        pad_left = (n_horz - data_horz)//2
        pad_right = n_horz - data_horz - pad_left
        if verbose:
            print("Warning: cropping %d pixels from top and bottom of the image" %crop_up)
        c_im = np.lib.pad(c_im[crop_up:-crop_down], ((0,0), (pad_left, pad_right)), 'constant', constant_values=0)

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

# The next two functions are adapted from
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to convert to polar coordinates.

def reproject_image_into_polar(data, origin=None):
    """Reprojects a 2D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) and defaults to the center of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2, ny//2)

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)
    r, theta = cart2polar(x, y)

    nr = r.max()
    nt = ny//2

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr)
    theta_i = np.linspace(theta.min(), theta.max(), nt)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    X += origin[0] # We need to shift the origin
    Y += origin[1] # back to the lower-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))
    return output, r_i, theta_i


def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the lower left corner of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_x, origin_y = origin
    x, y = np.meshgrid(np.arange(nx), np.arange(ny))
    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    """
    Transform carthesian coordinates to polar
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta


def polar2cart(r, theta):
    """
    Transform polar coordinates to carthesian
    """
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    return x, y
