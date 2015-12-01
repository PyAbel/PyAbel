# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .tools import get_image_quadrants


def is_symmetric(arr, i_sym=True, j_sym=True):
    """
    Takes in an array of shape (n, m) and check if it is symmetric

    Parameters:
       - arr: 1D or 2D array
       - i_sym: array is symmetric with respect to the 1st axis
       - j_sym: array is symmetric with respect to the 2nd axis

    Returns:
       a binary array with the symmetry condition for the corresponding quadrants.
       The global validity can be checked with `array.all()`

    Note: if both i_sym=True and i_sym=True, the input array is checked
    for polar symmetry.

    See https://github.com/PyAbel/PyAbel/issues/34#issuecomment-160344809 for
    the defintion of a center of the image.
    """

    Q0, Q1, Q2, Q3 = get_image_quadrants(arr)


    if i_sym and not j_sym:
        valid_flag = [ np.allclose(np.fliplr(Q1), Q0),
                       np.allclose(np.fliplr(Q2), Q3) ]
    elif not i_sym and j_sym:
        valid_flag = [ np.allclose(np.flipud(Q1), Q2),
                       np.allclose(np.flipud(Q0), Q3) ]
    elif i_sym and j_sym:
        valid_flag = [ np.allclose(np.flipud(np.fliplr(Q1)), Q3),
                       np.allclose(np.flipud(np.fliplr(Q0)), Q2) ]
    else:
        raise ValueError('Checking for symmetry with both i_sym=False and j_sym=False'\
                         'does not make sens!') 

    return np.array(valid_flag)



def absolute_ratio_benchmark(analytical, recon):
    """
    Check the absolute ratio between an analytical function and the result
     of a inv. Abel reconstruction.

    Parameters
    ----------
      - analytical: one of the classes from abel.analytical, initialized
      - recon: 1D ndarray: a reconstruction (i.e. inverse abel) given by some PyAbel implementation
    """
    mask = analytical.mask_valid
    err = analytical.func[mask]/recon[mask]
    return np.mean(err), np.std(err), np.sum(mask)

