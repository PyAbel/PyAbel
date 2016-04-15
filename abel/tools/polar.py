# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize


def reproject_image_into_polar(data, origin=None, Jacobian=False,
                               dr=1, dt=None):
    """
    Reprojects a 2D numpy array (``data``) into a polar coordinate system.
    "origin" is a tuple of (x0, y0) relative to the bottom-left image corner,
    and defaults to the center of the image.

    Parameters
    ----------
    data : 2D np.array
    origin : tuple
        The coordinate of the image center, relative to bottom-left
    Jacobian : boolean
        Include ``r`` intensity scaling in the coordinate transform. 
        This should be included to account for the changing pixel size that 
        occurs during the transform.
    dr : float
        Radial coordinate spacing for the grid interpolation
        tests show that there is not much point in going below 0.5
    dt : float
        Angular coordinate spacing (in degrees)
        if ``dt=None``, dt will be set such that the number of theta values
        is equal to the maximum value between the height or the width of 
        the image.

    Returns
    -------
    output : 2D np.array
        The polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of theta coordinates
        
    Notes
    -----
    Adapted from: 
    http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
    
    """
    # bottom-left coordinate system requires numpy image to be np.flipud
    data = np.flipud(data)

    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2 + nx % 2, ny//2 + ny % 2)   # % handles odd size image

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = np.round((r.max()-r.min())/dr)

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in degrees
        nt = np.round((theta.max()-theta.min())/(np.pi*dt/180))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)

    X += origin[0]  # We need to shift the origin
    Y += origin[1]  # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((yi, xi))  # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output = output*r_i[:, np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin=None):
    """
    Creates x & y coords for the indicies in a numpy array
    
    Parameters
    ----------
    data : numpy array
        2D data
    origin : (x,y) tuple
        defaults to the center of the image. Specify origin=(0,0)
        to set the origin to the *bottom-left* corner of the image.
    
    Returns
    -------
        x, y : arrays
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx//2+nx % 2, ny//2+ny % 2   # % for odd-size
    else:
        origin_x, origin_y = origin

    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))

    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar
    
    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    
    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates
        
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta


def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian
    
    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates
        
    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)   # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y
