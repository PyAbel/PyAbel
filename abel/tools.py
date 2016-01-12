# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage.interpolation import shift
from scipy.optimize import curve_fit, minimize 

def calculate_speeds(IM, origin=None, Jacobian=False, dr=1, dt=None):
    """ This performs an angular integration of the image returning 
        the one-dimentional intensity profile as a function of the 
        radial coordinate. 
        
     Parameters
     ----------
      - IM: a rows x cols ndarray.
      - origin: tuple, image center coordinate relative to *bottom-left* corner
                defaults to (rows//2+rows%2,cols//2+cols%2)
      - Jacobian: boolean, include r*sinθ in the angular sum (integration)
      - dr: float, radial coordinate grid spacing, in pixels (default 1)
      - dt: float, theta coordinate grid spacing in degrees
            (default rows//2)
      
     Returns
     -------
      - speeds: a 1D array of the integrated intensity 
      - r: the 1D array of radial coordinates.
      - theta: 1D array of theta coordinates
     """
    
    polarIM, r_grid, theta_grid = reproject_image_into_polar(IM, origin, 
                                       Jacobian=Jacobian, dr=dr, dt=dt)
    theta = theta_grid[0,:]  # theta coordinates
    r = r_grid[:,0]          # radial coordinates

    if Jacobian:  #  x r sinθ    
        sintheta = np.abs(np.sin(theta))
        polarIM = polarIM*sintheta[np.newaxis, :]
        polarIM = polarIM*r[:,np.newaxis]
    
    speeds = np.sum(polarIM, axis=1)
    
    return speeds, r[:IM.shape[1]]   # width = image width


def calculate_angular_distributions(IM, radial_ranges=None):
    """
    Intensity variation in the angular coordinate, theta

    This function is the theta-coordinate complement to 'calculate_speeds(IM)'

    (optionally and more useful) returning intensity vs angle for specific
    radial ranges

    Parameters
    ----------
     - IM: rows x cols  numpy array - image

     - radial_ranges: tuple list [(r0, r1), (r2, r3), ...] 
                      Evaluate the intensity vs angle for the radial
                      ranges r0_r1, r2_r3, etc. 

    Returns
    ---------
     - theta: 1d numpy array of angles, relative to vertical direction
     - angular_dist: cols/2 x number of radial ranges numpy array 
                     Intensity vs angle distribution for each selected
                     radial range
    """

    polarIM, r_grid, theta_grid = reproject_image_into_polar(IM)

    theta = theta_grid[0,:]  # theta coordinates
    r = r_grid[:,0]          # radial coordinates

    if radial_ranges is None:
        radial_ranges = [(0,r[-1]),]

    intensity_vs_theta_at_R = []
    for rr in radial_ranges:
        subr = np.logical_and(r >= rr[0],r <= rr[1])

        # sum intensity across radius of spectral feature
        intensity_vs_theta_at_R.append(np.sum(polarIM[subr], axis=0))

    return theta, intensity_vs_theta_at_R


def anisotropy_parameter(theta, intensity, theta_ranges=None):
    """
    Evaluate anisotropy parameter ß, for I vs θ data
    
         I = σ_total/4π [ 1 + β P2(cosθ) ]     Eq.(1)

     where P2(x)=(3x^2-1)/2 is a 2nd order Legendre polynomial.
    
    Cooper and Zare "Angular distribution of photoelectrons"
    J Chem Phys 48, 942-943 (1968) doi:10.1063/1.1668742


    Parameters:
    -----------
     - theta
     - intensity: 1d numpy array
     - theta_ranges: list of tuple angular ranges over which to fit
                     
    Returns:
    --------
      - fit parameters: (beta, error_beta), (amplitude, error_amplitude)

    """
    def P2(x):   # 2nd order Legendre polynomial
        return (3*x*x-1)/2

    def PAD(theta, beta, amplitude):
        return amplitude*(1 + beta*P2(np.cos(theta)))   # Eq. (1) as above

    # select data to be included in the fit by θ
    if theta_ranges is not None:
        subtheta = np.ones(len(theta), dtype=bool) 
        for rt in theta_ranges:
            subtheta = np.logical_and(subtheta==True,
                       np.logical_and(theta >= rt[0],theta <= rt[1]))
        theta = theta[subtheta]
        intensity = intensity[subtheta] 

    # fit angular intensity distribution 
    popt, pcov = curve_fit(PAD, theta, intensity)

    beta, amplitude = popt
    error_beta, error_amplitude = np.sqrt(np.diag(pcov))

    return (beta, error_beta), (amplitude, error_amplitude)

def get_image_quadrants(IM, reorient=False):
    """
    Given an image (m,n) return its 4 quadrants Q0, Q1, Q2, Q3
    as defined in abel.hansenlaw.iabel_hansenlaw

    Parameters:
      - IM: 1D or 2D array
      - reorient: reorient image as required by abel.hansenlaw.iabel_hansenlaw
    """
    IM = np.atleast_2d(IM)

    n, m = IM.shape

    n_c = n//2 + n%2
    m_c = m//2 + m%2

    # define 4 quadrants of the image
    # see definition in abel.hansenlaw.iabel_hansenlaw
    Q1 = IM[:n_c, :m_c]
    Q2 = IM[-n_c:, :m_c]
    Q0 = IM[:n_c, -m_c:]
    Q3 = IM[-n_c:, -m_c:]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))

    return Q0, Q1, Q2, Q3

def put_image_quadrants (Q,odd_size=True):
    """
    Reassemble image from 4 quadrants Q = (Q0, Q1, Q2, Q3)
    The reverse process to get_image_quadrants()
    Qi defined in abel.hansenlaw.iabel_hansenlaw
    
    Parameters:
      - Q: tuple of numpy array quadrants
      - even_size: boolean, whether final image is even or odd pixel size
                   odd size requires trimming 1 row from Q1, Q0, and
                                              1 column from Q1, Q2

    Returns:  
      - rows x cols numpy array - the reassembled image
    """


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


def center_image_by_slice(IM, slice_width=10, radial_range=(0,-1),
                          center_vertical=True, center_horizontal=True,
                          pixel_center=True):
    """
    Center image by comparing opposite side, vertical and
    horizontal slice profiles 

    Parameters
    ----------
      - IM : rows x cols numpy array 
      
      - slice_width : add together this number of rows (cols) to improve signal
      
      - radial_range(rmin,rmax): radial range [rmin,rmax] for slide profile comparison
      - center_vertical: boolean, find vertical center
      - center_horizontal: boolean, find horizontal center
      - pixel_center: boolean, make center of image within a pixel

    """
    # intensity difference between an axial slice and its shifted opposite
    def align(offset, top_or_left_slice, bottom_or_right_slice):

        diff = shift(top_or_left_slice,offset) - bottom_or_right_slice

        return (diff**2).sum()

    rows,cols = IM.shape
    r2 = rows//2
    c2 = cols//2
    sw2 = slice_width//2

    # slices - sum across slice_width rows (cols)
    left  = IM[r2-sw2:r2+sw2, :c2].sum(axis=0)
    right = IM[r2-sw2:r2+sw2, c2:].sum(axis=0)
    top   = IM[:r2, c2-sw2:c2+sw2].sum(axis=1)
    bot   = IM[r2:, c2-sw2:c2+sw2].sum(axis=1)

    # reorient slices to the same direction, select region [rmin:rmax]
    rmin, rmax = radial_range
    left  = left[::-1][rmin:rmax]   # flip same direction as 'right'
    right = right[rmin:rmax]
    top   = top[::-1][rmin:rmax]    # flip
    bot   = bot[rmin:rmax]

    # compare and determine shift to best overlap slice profiles
    # fix me! - should be consistent and use curve_fit()?
    shift0 = [0,]
    if center_horizontal:
        horiz = minimize(align, shift0, args=(left, right))
    else: 
        horiz['x'] = 0.0
   
    if center_vertical:
        vert  = minimize(align, shift0, args=(top, bot))
    else:
        vert['x'] = 0.0

    col_shift = horiz['x']
    row_shift = vert['x']

    if pixel_center: 
        col_shift += 0.5
        row_shift -= 0.5

    IM_centered = shift(IM,(row_shift,col_shift)) # center image

    if rows%2==0 and pixel_center:    # make odd size image
        IM_centered = IM_centered[:-1, 1:]  # drop left most column, bottom row

    return IM_centered, (row_shift,col_shift)


# The next two functions are adapted from
# http://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system
# It is possible that there is a faster way to convert to polar coordinates.

def reproject_image_into_polar(data, origin=None, Jacobian=False,
                               dr=1, dt=None):
    """Reprojects a 2D numpy array ("data") into a polar coordinate system.
    "origin" is a tuple of (x0, y0) relative to the bottom-left image corner,
    and defaults to the center of the image.
    
    Parameters
    ----------
     - data:   rowsxcolums numpy array
     - origin: tuple, the coordinate of the image center, relative to bottom-left
     - Jacobian: boolean, include r intensity scaling in the coordinate transform
     - dr: radial coordinate spacing for the grid interpolation
             tests show that there is not much point in going below 0.5
     - dt: angular coordinate spacing (in degrees)

     Returns
     -------
      - output:     rows x cols  or row(col)xrow(col) numpy array, polar image
      - r_grid:     meshgrid of radial coordinates
      - theta_grid: meshgrid of theta coordinates
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin = (nx//2+nx%2, ny//2+ny%2)   # % handles odd size image

    # Determine that the min and max r and theta coords will be...
    x, y = index_coords(data, origin=origin)  # (x,y) coordinates of each pixel
    r, theta = cart2polar(x, y)               # convert (x,y) -> (r,θ)
                                              # note θ=0 is vertical

    nr = np.round((r.max()-r.min())/dr)
         
    if dt is None:
       nt = ny
    else:
       nt = np.round((theta.max()-theta.min())/(np.pi*dt/180))  # dt in degrees

    # Make a regular (in polar space) grid based on the min and max r & theta
    r_i     = np.linspace(r.min(), r.max(), nr, endpoint=False)
    theta_i = np.linspace(theta.min(), theta.max(), nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)

    # Project the r and theta grid back into pixel coordinates
    X, Y = polar2cart(r_grid, theta_grid)

    X += origin[0] # We need to shift the origin
    Y += origin[1] # back to the bottom-left corner...
    xi, yi = X.flatten(), Y.flatten()
    coords = np.vstack((xi, yi)) # (map_coordinates requires a 2xn array)

    zi = map_coordinates(data, coords)
    output = zi.reshape((nr, nt))

    if Jacobian:
        output = output*r_i[:,np.newaxis]

    return output, r_grid, theta_grid


def index_coords(data, origin=None):
    """Creates x & y coords for the indicies in a numpy array "data".
    "origin" defaults to the center of the image. Specify origin=(0,0)
    to set the origin to the *bottom-left* corner of the image.
    """
    ny, nx = data.shape[:2]
    if origin is None:
        origin_x, origin_y = nx//2+nx%2, ny//2+ny%2   # % to handle odd-size
    else:
        origin_x, origin_y = origin
    
    x, y = np.meshgrid(np.arange(float(nx)), np.arange(float(ny)))
    
    x -= origin_x
    y -= origin_y
    return x, y


def cart2polar(x, y):
    """
    Transform carthesian coordinates to polar
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta 


def polar2cart(r, theta):
    """
    Transform polar coordinates to carthesian
    """
    y = r * np.sin(theta)   # θ referenced to vertical
    x = r * np.cos(theta)
    return x, y



class CythonExtensionsNotBuilt(Exception):
    pass


CythonExtensionsNotBuilt_msg = CythonExtensionsNotBuilt(
        "Cython extensions were not propery built.\n"
        "Either the complilation failed at the setup phase (no complier, compiller not found etc),\n"
        "or you are using Windows 64bit with Anaconda that has a known issue with Cython\n"
        "https://groups.google.com/a/continuum.io/forum/#!topic/anaconda/3ES7VyW4t3I \n"
        )

