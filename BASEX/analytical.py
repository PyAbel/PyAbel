# -*- coding: utf-8 -*-
import numpy as np

def abel_step_analytical(r, A0, r0, r1):
    """
    Direct Abel transform of a step function located between r0 and r1,
    with a height A0

     A0 +                  +-------------+
        |                  |             |
        |                  |             |
      0 | -----------------+             +-------------
        +------------------+-------------+------------>
        0                  r0            r1           r axis

    This function is mostly used for unit testing the inverse Abel transform

    Parameters:

       r:   1D array, vecor of positions along the r axis. Must start with 0.
       r0, r1: floats, positions of the step along the r axis
       A0:  float or 1D array: height of the step. If 1D array the height can be
            variable along the Z axis

    Returns:
       1D array if A0 is a float, a 2D array otherwise
    """

    if r[0] != 0.0:
        raise ValueError('The vector of r coordinates must start with 0.0')

    F_1d = np.zeros(r.shape)
    mask = (r>=r0)*(r<r1)
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2)
    mask = r<r0
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2) - 2*np.sqrt(r0**2 - r[mask]**2)
    A0 = np.atleast_1d(A0)
    A0 = A0[:, np.newaxis]

    return F_1d*A0


def sym_abel_step_1d(r, A0, r0, r1):
    """
    Produces a symmetrical analytical transform of a 1d step
    """
    d = np.empty(r.shape)
    for sens, mask in enumerate([r>=0, r<=0]):
        d[mask,:] =  abel_step_analytical(np.abs(r[mask]), A0, r0, r1)

    return d
