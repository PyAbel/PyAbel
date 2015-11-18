# -*- coding: utf-8 -*-
import numpy as np


class SymStep(object):
    def __init__(self, n, r_max, r1, r2, A0=1.0):
        """
        Define a a symmetric step function and calculate it's analytical
        Abel transform. See examples/example_step.py

        Parameters
           - n : int: number of points along the r axis
           - r_max: float: range of the symmetric r interval
           - r1, r2: floats: bounds of the step function if r > 0
                    ( symetic function is constructed for r < 0)
           - A0: float: height of the step
        """

        self.n = n
        self.r_max = r_max
        self.r1, self.r2 = r1, r2
        self.A0 = A0

        self.r = r = np.linspace(-r_max, r_max, n)
        mask = np.abs(np.abs(r)- 0.5*(r1 + r2)) < 0.5*(r2 - r1)
        self.dr =  np.diff(r)[0]
        fr = np.zeros(r.shape)
        fr[mask] = A0
        self.func = fr

    @property
    def abel(self):
        """ Return the direct Abel transform """
        return sym_abel_step_1d(self.r, self.A0, self.r1, self.r2)[0]


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

    if np.all(r[[0,-1]]) :
        raise ValueError('The vector of r coordinates must start with 0.0')

    F_1d = np.zeros(r.shape)
    mask = (r>=r0)*(r<r1)
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2)
    mask = r<r0
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2) - 2*np.sqrt(r0**2 - r[mask]**2)
    A0 = np.atleast_1d(A0)
    if A0.ndim == 1:
        A0 = A0[:,np.newaxis]
    return F_1d[np.newaxis,:]*A0


def sym_abel_step_1d(r, A0, r0, r1):
    """
    Produces a symmetrical analytical transform of a 1d step
    """
    A0 = np.atleast_1d(A0)
    d = np.empty(A0.shape + r.shape)
    A0 = A0[np.newaxis, :]
    for sens, mask in enumerate([r>=0, r<=0]):
        d[:,mask] =  abel_step_analytical(np.abs(r[mask]), A0, r0, r1)

    return d
