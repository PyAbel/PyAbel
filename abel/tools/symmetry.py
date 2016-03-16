# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import warnings
from scipy import fftpack


def get_image_quadrants(IM, reorient=True, symmetry_axis=None,
                        use_quadrants=(True, True, True, True),
                        symmetrize_method="average"):
    """
    Given an image (m,n) return its 4 quadrants Q0, Q1, Q2, Q3
    as defined below.

    Parameters
    ----------
    IM : 2D np.array
        Image data shape (rows, cols)

    reorient : boolean
        Reorient quadrants to match the orientation of Q0 (top-right)

    symmetry_axis : int or tuple
        can have values of ``None``, ``0``, ``1``, or ``(0,1)`` and specifies 
        no symmetry, vertical symmetry axis, horizontal symmetry axis, and both vertical
        and horizontal symmetry axes. Quadrants are added.
        See Note. 

    use_quadrants : boolean tuple
       Include quadrant (Q0, Q1, Q2, Q3) in the symmetry combination(s)
       and final image

    symmetrize_method: str
       Method used for symmetrizing the image.      
 
       average: Simply average the quadrants.
       fourier: Axial symmetry implies that the Fourier components of the 2-D 
                projection should be real. Removing the imaginary components in 
                reciprocal space leaves a symmetric projection.
                ref: Overstreet, K., et al. "Multiple scattering and the density 
                     distribution of a Cs MOT." Optics express 13.24 (2005): 9672-9682.
                     http://dx.doi.org/10.1364/OPEX.13.009672

    Returns
    -------
    Q0, Q1, Q2, Q3 : tuple of 2D np.arrays
      shape: (``rows//2+rows%2, cols//2+cols%2``)
      all oriented in the same direction as Q0 if ``reorient=True``

    
    Notes
    -----
     
    The symmetry_axis keyword averages quadrants like this: ::

         +--------+--------+
         | Q1   * | *   Q0 |
         |   *    |    *   |
         |  *     |     *  |               cQ1 | cQ0
         +--------o--------+ --(output) -> ----o----
         |  *     |     *  |               cQ2 | cQ3
         |   *    |    *   |
         | Q2  *  | *   Q3 |          cQi == combined quadrants
         +--------+--------+                 

        symmetry_axis = None - individual quadrants
        symmetry_axis = 0 (vertical) - average Q0+Q1, and Q2+Q3
        symmetry_axis = 1 (horizontal) - average Q1+Q2, and Q0+Q3
        symmetry_axis = (0, 1) (both) - combine and average all 4 quadrants
    

    The end results look like this: ::
    
        (0) symmetry_axis = None
        
            returned image   Q1 | Q0
                            ----o----
                             Q2 | Q3
                        
        (1) symmetry_axis = 0

            Combine:  Q01 = Q0 + Q1, Q23 = Q2 + Q3
            returned image    Q01 | Q01
                             -----o-----
                              Q23 | Q23
                          
        (2) symmetry_axis = 1

            Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
            returned image   Q12 | Q03
                            -----o-----
                             Q12 | Q03

        (3) symmetry_axis = (0, 1)

            Combine all quadrants: Q = Q0 + Q1 + Q2 + Q3
            returned image   Q | Q
                            ---o---  all quadrants equivalent
                             Q | Q

    """
    IM = np.atleast_2d(IM)

    if not isinstance(symmetry_axis, (list, tuple)):
        # if the user supplies an int, make it into a 1-element list:
        symmetry_axis = [symmetry_axis]

    if ((symmetry_axis == [None] and (use_quadrants[0]==False
                                   or use_quadrants[1]==False 
                                   or use_quadrants[2]==False
                                   or use_quadrants[3]==False)) or 
        # at least one empty
        (symmetry_axis == [0] and use_quadrants[0]==False and 
                                  use_quadrants[1]==False) or # top empty
        (symmetry_axis == [0] and use_quadrants[2]==False and 
                                  use_quadrants[3]==False) or # bot empty
        (symmetry_axis == [1] and use_quadrants[1]==False and
                                  use_quadrants[2]==False) or # left empty
        (symmetry_axis == [1] and use_quadrants[0]==False and 
                                  use_quadrants[3]==False)    # right empty
                                                                or
        not np.any(use_quadrants)
        ): 
        raise ValueError('At least one quadrant would be empty.'
                         ' Please check symmetry_axis and use_quadrant'
                         ' values to ensure that all quadrants will have a'
                         ' defined value.')

    n, m = IM.shape

    # odd size increased by 1
    n_c = n // 2 + n % 2
    m_c = m // 2 + m % 2
    

    if isinstance(symmetry_axis, tuple) and not reorient:
        raise ValueError(
            'In order to add quadrants (i.e., to apply horizontal or \
            vertical symmetry), you must reorient the image.')

    if symmetrize_method == "fourier":
        if np.sum(use_quadrants)<4:
            warnings.warn("Using Fourier transformation to symmetrize the"
                          " data will use all 4 quadrants!!")
        if 0 in symmetry_axis:
            IM = fftpack.ifft(fftpack.fft(IM).real).real
        if 1 in symmetry_axis:
            IM = fftpack.ifft(fftpack.fft(IM.T).real).T.real

    # define 4 quadrants of the image
    # see definition above
    Q0 = IM[:n_c, -m_c:]*use_quadrants[0]
    Q1 = IM[:n_c, :m_c]*use_quadrants[1]
    Q2 = IM[-n_c:, :m_c]*use_quadrants[2]
    Q3 = IM[-n_c:, -m_c:]*use_quadrants[3]

    if reorient:
        Q1 = np.fliplr(Q1)
        Q3 = np.flipud(Q3)
        Q2 = np.fliplr(np.flipud(Q2))
    
    if symmetrize_method == "fourier":
        return Q0, Q1, Q2, Q3
  
    elif symmetrize_method == "average": 
        if symmetry_axis==(0, 1):
            Q = (Q0 + Q1 + Q2 + Q3)/np.sum(use_quadrants)
            return Q, Q, Q, Q

        if 0 in symmetry_axis:   #  vertical axis image symmetry
            Q0 = Q1 = (Q0 + Q1)/(use_quadrants[0] + use_quadrants[1])
            Q2 = Q3 = (Q2 + Q3)/(use_quadrants[2] + use_quadrants[3])

        if 1 in symmetry_axis:   # horizontal axis image symmetry
            Q1 = Q2 = (Q1 + Q2)/(use_quadrants[1] + use_quadrants[2])
            Q0 = Q3 = (Q0 + Q3)/(use_quadrants[0] + use_quadrants[3])

        return Q0, Q1, Q2, Q3
    else:
        raise ValueError("Invalid method for symmetrizing the image!!")


def put_image_quadrants(Q, original_image_shape, symmetry_axis=None):
    """
    Reassemble image from 4 quadrants Q = (Q0, Q1, Q2, Q3)
    The reverse process to get_image_quadrants(reorient=True)

    Note: the quadrants should all be oriented as Q0, the upper right quadrant

    Parameters
    ----------
    Q: tuple of np.array  (Q0, Q1, Q2, Q3)
       Image quadrants all oriented as Q0
       shape (``rows//2+rows%2, cols//2+cols%2``) ::

            +--------+--------+
            | Q1   * | *   Q0 |
            |   *    |    *   |
            |  *     |     *  |
            +--------o--------+ 
            |  *     |     *  |  
            |   *    |    *   |
            | Q2  *  | *   Q3 | 
            +--------+--------+                 

    original_image_shape: tuple
       (rows, cols)

       reverses the padding added by `get_image_quadrants()` for odd-axis sizes

       odd row trims 1 row from Q1, Q0

       odd column trims 1 column from Q1, Q2

    symmetry_axis : int or tuple
       impose image symmetry
       
           ``symmetry_axis = 0 (vertical)   - Q0 == Q1 and Q3 == Q2``
           ``symmetry_axis = 1 (horizontal) - Q2 == Q1 and Q3 == Q0``


    Returns
    -------
    IM : np.array
        Reassembled image of shape (rows, cols): ::
        
            symmetry_axis =

             None             0              1           (0,1)

            Q1 | Q0        Q1 | Q1        Q1 | Q0       Q1 | Q1
           ----o----  or  ----o----  or  ----o----  or ----o----
            Q2 | Q3        Q2 | Q2        Q1 | Q0       Q1 | Q1

    """

    Q0, Q1, Q2, Q3 = Q

    if not isinstance(symmetry_axis, (list, tuple)):
        # if the user supplies an int, make it into a 1-element list:
        symmetry_axis = [symmetry_axis]

    if 0 in symmetry_axis:
        Q0 = Q1
        Q3 = Q2

    if 1 in symmetry_axis:
        Q2 = Q1
        Q3 = Q0

    if original_image_shape[0] % 2:
        # odd-rows => remove duplicate bottom row from Q1, Q0
        Q0 = Q0[:-1, :]
        Q1 = Q1[:-1, :]

    if original_image_shape[1] % 2:
        # odd-columns => remove duplicate first column from Q1, Q2
        Q1 = Q1[:, 1:]
        Q2 = Q2[:, 1:]

    Top = np.concatenate((np.fliplr(Q1), Q0), axis=1)
    Bottom = np.flipud(np.concatenate((np.fliplr(Q2), Q3), axis=1))

    IM = np.concatenate((Top, Bottom), axis=0)

    return IM


# def center_image(data, center, n, ndim=2):
#     """
#     This centers the image at the given center and makes it of size n by n
#
#     THIS FUNCTION IS DEPRECIATED.
#     All centering functions should be moves to abel.tools.center
#     """
#
#     Nh, Nw = data.shape
#     n_2 = n//2
#     if ndim == 1:
#         cx = int(center)
#         im = np.zeros((1, 2*n))
#         im[0, n-cx:n-cx+Nw] = data
#         im = im[:, n_2:n+n_2]
#         # This is really not efficient
#         # Processing 2D image with identical rows while we just want a
#         # 1D slice
#         im = np.repeat(im, n, axis=0)
#
#     elif ndim == 2:
#         cx, cy = np.asarray(center, dtype='int')
#
#         # Make an array of zeros that is large enough for cropping or padding:
#         sz = 2*np.round(n + np.max((Nw, Nh)))
#         im = np.zeros((sz, sz))
#
#         # Set center of "zeros image" to be the data
#         im[sz//2-cy:sz//2-cy+Nh, sz//2-cx:sz//2-cx+Nw] = data
#
#         # Crop padded image to size n
#         # note the n%2 which return the appropriate image size for both
#         # odd and even images
#         im = im[sz//2-n_2:n_2+sz//2+n % 2, sz//2-n_2:n_2+sz//2+n % 2]
#
#     else:
#         raise ValueError
#
#     return im
#
#
# def center_image_asym(data, center_column, n_vert, n_horz, verbose=False):
#     """
#     This centers a (rectangular) image at the given center_column
#     and makes it of size n_vert by n_horz
#
#     THIS FUNCTION IS DEPRECIATED.
#     All centering functions should be moved to abel.tools.center
#     """
#
#     if data.ndim > 2:
#         raise ValueError("Array to be centered must be 1- or 2-dimensional")
#
#     c_im = np.copy(data)  # make a copy of the original data for manipulation
#     data_vert, data_horz = c_im.shape
#     pad_mode = str("constant")
#
#     if data_horz % 2 == 0:
#         # Add column of zeros to the extreme right
#         # to give data array odd columns
#         c_im = np.pad(c_im, ((0, 0), (0, 1)), pad_mode, constant_values=0)
#         data_vert, data_horz = c_im.shape  # update data dimensions
#
#     delta_h = int(center_column - data_horz//2)
#     if delta_h != 0:
#         if delta_h < 0:
#             # Specified center is to the left of nominal center
#             # Add compensating zeroes on the left edge
#             c_im = np.pad(c_im, ((0, 0), (2*np.abs(delta_h), 0)), pad_mode,
#                           constant_values=0)
#             data_vert, data_horz = c_im.shape
#         else:
#             # Specified center is to the right of nominal center
#             # Add compensating zeros on the right edge
#             c_im = np.pad(c_im, ((0, 0), (0, 2*delta_h)), pad_mode,
#                           constant_values=0)
#             data_vert, data_horz = c_im.shape
#
#     if n_vert >= data_vert and n_horz >= data_horz:
#         pad_up = (n_vert - data_vert)//2
#         pad_down = n_vert - data_vert - pad_up
#         pad_left = (n_horz - data_horz)//2
#         pad_right = n_horz - data_horz - pad_left
#
#         c_im = np.pad(
#             c_im, ((pad_up, pad_down), (pad_left, pad_right)),
#             pad_mode, constant_values=0)
#
#     elif n_vert >= data_vert and n_horz < data_horz:
#         pad_up = (n_vert - data_vert)//2
#         pad_down = n_vert - data_vert - pad_up
#         crop_left = (data_horz - n_horz)//2
#         crop_right = data_horz - n_horz - crop_left
#         if verbose:
#             print("Warning: cropping %d pixels from the sides \
#                    of the image" % crop_left)
#         c_im = np.pad(
#             c_im[:, crop_left:-crop_right], ((pad_up, pad_down), (0, 0)),
#             pad_mode, constant_values=0)
#
#     elif n_vert < data_vert and n_horz >= data_horz:
#         crop_up = (data_vert - n_vert)//2
#         crop_down = data_vert - n_vert - crop_up
#         pad_left = (n_horz - data_horz)//2
#         pad_right = n_horz - data_horz - pad_left
#         if verbose:
#             print("Warning: cropping %d pixels from top and bottom \
#                    of the image" % crop_up)
#         c_im = np.pad(
#             c_im[crop_up:-crop_down], ((0, 0), (pad_left, pad_right)),
#             pad_mode, constant_values=0)
#
#     elif n_vert < data_vert and n_horz < data_horz:
#         crop_up = (data_vert - n_vert)//2
#         crop_down = data_vert - n_vert - crop_up
#         crop_left = (data_horz - n_horz)//2
#         crop_right = data_horz - n_horz - crop_left
#         if verbose:
#             print("Warning: cropping %d pixels from top and bottom \
#                    and %d pixels from the sides of the image " % (
#                     crop_up, crop_left))
#         c_im = c_im[crop_up:-crop_down, crop_left:-crop_right]
#
#     else:
#         raise ValueError('Input data dimensions incompatible \
#                           with chosen basis set.')
#
#     return c_im
