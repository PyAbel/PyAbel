# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates

def calculate_speeds(IM, n):
    # This section is to get the speed distribution.
    # The original matlab version used an analytical formula to get the speed distribution directly
    # from the basis coefficients. But, the C version of BASEX uses a numerical method similar to
    # the one implemented here. The difference between the two methods is negligable.
    """ Generating the speed distribution """


    nx,ny = np.shape(IM)
    xi = np.linspace(-100, 100, nx)
    yi = np.linspace(-100, 100, ny)
    X,Y = np.meshgrid(xi, yi)

    polarIM, ri, thetai = reproject_image_into_polar(IM)

    speeds = np.sum(polarIM, axis=1)
    speeds = speeds[:n//2] #Clip off the corners

    return speeds


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
        
        #make an array of zeros that is large enough for cropping or padding:
        sz = 2*np.round(n + np.max((Nw, Nh)))
        im = np.zeros((sz, sz))
        im[sz//2-cy:sz//2-cy+Nh, sz//2-cx:sz//2-cx+Nw] = data
        
        if n%2==0:
            # If the image size is even, use lower left of the central 4 pixels as center
            im = im[ sz//2-n_2-1:n_2+sz//2, sz//2-n_2-1:n_2+sz//2]
        if n%2==1:
            # If the image size is odd, set the central pixel to be exactly center
            im = im[ sz//2-n_2:n_2+sz//2+1, sz//2-n_2:n_2+sz//2+1]
        
        #print(np.shape(im))
    else:
        raise ValueError

    return im





# I got these next two functions from a stackoverflow page and slightly modified them.
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to get the speed distribution.
# If you figure it out, pease let me know! (danhickstein@gmail.com)

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
