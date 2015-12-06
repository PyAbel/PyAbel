# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates

def calculate_speeds(IM,origin=None):
    """ This performs an angular integration of the image and returns the one-dimentional intensity profile 
        as a function of the radial coordinate. It assumes that the image is properly centered. 
        
     Parameters
     ----------
      - IM: a NxN ndarray.
      
     Returns
     -------
      - speeds: a 1D array of the integrated intensity versus the radial coordinate.
     """
    
    polarIM, ri, thetai = reproject_image_into_polar(IM,origin)

    speeds = np.sum(polarIM, axis=1)
    
    # Clip the data corresponding to the corners, since these pixels contain incomplete information
    n = np.min(np.shape(IM))//2     # find the shortest radial coordinate
    speeds = speeds[:n]          # clip the 1D data

    return speeds


def add_image_col (im):
    """
    Add 1 column at image centre
    increases image shape to (rows,columns+1)
    
    Basex basis calculation requires odd size image
    Hansen and Law operates on quadrants, an even size image
    """
    rows,cols = im.shape
    n2 = cols//2 + cols%2  # mid point
    # add centre row
    #im = np.concatenate((im[:n2],im[n2-1:n2],im[n2:]),axis=0)
    # add centre column
    im = np.concatenate((im[:,:n2],im[:,n2-1:n2],im[:,n2:]),axis=1)
    return im

def delete_image_col (im):
    """
    Delete 1 column at image centre
    decreases image shape to (rows,columns-1)
    
    Basex basis calculation requires odd size image width
    Hansen and Law splits image into quadrants => an even size image width
    """
    rows,cols = im.shape
    n2 = cols//2
    # delete centre row
    #im = np.concatenate((im[:n2],im[n2+1:]),axis=0)
    # delete centre column
    im = np.concatenate((im[:,:n2],im[:,n2+1:]),axis=1)
    return im
    
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
