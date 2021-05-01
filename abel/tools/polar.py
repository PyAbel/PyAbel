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
    Reprojects a 2D numpy array (**data**) into a polar coordinate system,
    with the pole placed at **origin** and the angle measured clockwise from
    the upward direction. The resulting array has rows corresponding to the
    radial grid, and columns corresponding to the angular grid.

    Parameters
    ----------
    data : 2D np.array
        the image array
    origin : tuple or None
        (row, column) coordinates of the image origin. If ``None``, the
        geometric center of the image is used.
    Jacobian : bool
        Include `r` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    dr : float
        radial coordinate spacing for the grid interpolation.
        Tests show that there is not much point in going below 0.5.
    dt : float or None
        angular coordinate spacing (in radians).
        If ``None``, the number of angular grid points will be set to the
        largest dimension (the height or the width) of the image.

    Returns
    -------
    output : 2D np.array
        the polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of angular coordinates

    Notes
    -----
    Adapted from:
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (ny // 2, nx // 2)
    else:
        origin = list(origin)
        # wrap negative coordinates
        if origin[0] < 0:
            origin[0] += ny
        if origin[1] < 0:
            origin[1] += nx

    # Determine what the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)  # convert (x,y) -> (r,θ), note θ=0 is vertical

    nr = int(np.ceil((r.max() - r.min()) / dr))

    if dt is None:
        nt = max(nx, ny)
    else:
        # dt in radians
        nt = int(np.ceil((theta.max() - theta.min()) / dt))

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Convert the r and theta grids to Cartesian coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    # then to a 2×n array of row and column indices for np.map_coordinates()
    rowi = (origin[0] - Y).flatten()
    coli = (X + origin[1]).flatten()
    coords = np.vstack((rowi, coli))

    # Remap with interpolation
    # (making an array of floats even if the data has an integer type)
    zi = map_coordinates(data, coords, output=float)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output *= r_i[:, np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin=None):
    """
    Creates `x` and `y` coordinates for the indices in a numpy array, relative
    to the **origin**, with the `x` axis going to the right, and the `y` axis
    going `up`.

    Parameters
    ----------
    data : numpy array
        2D data. Only the array shape is used.
    origin : tuple or None
        (row, column). Defaults to the geometric center of the image.

    Returns
    -------
        x, y : 2D numpy arrays
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx // 2, ny // 2
    else:
        origin_y, origin_x = origin
        # wrap negative coordinates
        if origin_y < 0:
            origin_y += ny
        if origin_x < 0:
            origin_x += nx

    x, y = np.meshgrid(np.arange(float(nx)) - origin_x,
                       origin_y - np.arange(float(ny)))
    return x, y


def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar.

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
    Transform polar coordinates to Cartesian.

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
