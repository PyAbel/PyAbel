# -*- coding: utf-8 -*-
import numpy as np
from abel.tools.polar import index_coords, cart2polar


# This file includes functions that have a known analytical Abel transform.
# They are used in unit testing as well for comparing different Abel impementations.
# See BaseAnalytical class for more information.

def sample_image_dribinski(n=361, origin=None):
    """
    Sample test image used in the BASEX paper Rev. Sci. Instrum. 73, 2634 (2002) 
    9x Gaussian functions of half-width 4 pixel + 1 background width 3600
      - anisotropy - ß = -1  for cosθ term
                   - ß = +2  for sinθ term
                   - ß =  0  isotropic, no angular variation
    (there are some missing negative exponents in the publication)

    Parameters
    ----------
     - n: integer (square) image width (height)
     - origin: tuple,  (row, col) center of image

    Returns
    -------
     - IM: 2D n x n numpy array image

    """

    def Gauss (r, r0, sigma2):
        return np.exp(-(r-r0)**2/sigma2)

    def I(r, theta):  # intensity function Eq. (16)
        sinetheta2 = np.sin(theta)**2
        cosinetheta2 = np.cos(theta)**2

        t0 = 7*Gauss(r, 10, 4)*sinetheta2
        t1 = 3*Gauss(r, 15, 4)
        t2 = 5*Gauss(r, 20, 4)*cosinetheta2
    
        t3 =   Gauss(r, 70, 4)
        t4 = 2*Gauss(r, 85, 4)*cosinetheta2
        t5 =   Gauss(r, 100, 4)*sinetheta2

        t6 = 2*Gauss(r, 145, 4)*sinetheta2
        t7 =   Gauss(r, 150, 4)
        t8 = 3*Gauss(r, 155, 4)*cosinetheta2

        t9 = 20*Gauss(r, 45, 3600)  # background under t3 to t5
    
        return 2000*(t0+t1+t2) + 200*(t3+t4+t5) + 50*(t6+t7+t8) + t9

    # meshgrid @DanHickstein issue #67

    x = np.arange(n)
    n2 = n//2 + n%2
    X, Y = np.meshgrid(x-n2, x-n2)
    R, THETA = cart2polar(X, Y)
    IM = I(R, THETA)

    return IM

class BaseAnalytical(object):

    def __init__(self, n, r_max, symmetric=True, **args):
        """This is the base class for functions that have a known Abel transform. 
        Every such class should expose the following public attributes:
          - self.r: vector of positions along the r axis
          - self.func: the values of the original function (same shape as self.r)
          - self.abel: the values of the Abel transform (same shape as self.r)
          - self.mask_valid: mask of the r interval where the function is well smoothed/well behaved 
                           (no known artefacts in the inverse Abel reconstuction), typically excluding
                           the origin, the domain boundaries, and function discontinuities, that can 
                           be used for unit testing. 

        See GaussianAnalytical for a concrete example.

        Parameters
        ----------
           - n : int: number of points along the r axis
           - r_max: float: maximum r interval
          - symmetric: if True the r interval is [-r_max, r_max]  (and n should be odd)
                       otherwise the r interval is [0, r_max]
        """
        self.n = n
        self.r_max = r_max

        assert r_max > 0

        if symmetric:
            self.r = np.linspace(-r_max, r_max, n)
        else:
            self.r = np.linspace(0, r_max, n)

        self.dr =  np.diff(self.r)[0]



class StepAnalytical(BaseAnalytical):
    def __init__(self, n, r_max, r1, r2, A0=1.0,
                                                ratio_valid_step=1.0, symmetric=True):
        """
        Define a a symmetric step function and calculate it's analytical
        Abel transform. See examples/example_step.py

        Parameters
           - n : int: number of points along the r axis
           - r_max: float: range of the symmetric r interval
           - symmetric: if True the r interval is [-r_max, r_max]  (and n should be odd)
                       otherwise the r interval is [0, r_max]
           - r1, r2: floats: bounds of the step function if r > 0
                    ( symetic function is constructed for r < 0)
           - A0: float: height of the step
           - ratio_valid_step: float: in the benchmark take only the central ratio*100% of the step
                                         (exclude possible artefacts on the edges)

        see https://github.com/PyAbel/PyAbel/pull/16

        """

        super(StepAnalytical, self).__init__(n, r_max, symmetric)

        self.r1, self.r2 = r1, r2
        self.A0 = A0

        mask = np.abs(np.abs(self.r)- 0.5*(r1 + r2)) < 0.5*(r2 - r1)

        fr = np.zeros(self.r.shape)
        fr[mask] = A0

        self.func = fr

        self.abel = sym_abel_step_1d(self.r, self.A0, self.r1, self.r2)[0]

        # exclude the region near the discontinuity
        self.mask_valid = np.abs(np.abs(self.r)- 0.5*(r1 + r2)) < \
                                                ratio_valid_step*0.5*(r2 - r1)



class GaussianAnalytical(BaseAnalytical):
    def __init__(self, n, r_max, sigma=1.0, A0=1.0,
                            ratio_valid_sigma=2.0, symmetric=True):
        """
        Define a gaussian function and calculate its analytical
        Abel transform. See examples/example_gaussian.py

        Parameters
           - n : int: number of points along the r axis
           - r_max: float: range of the symmetric r interval
           - symmetric: if True the r interval is [-r_max, r_max]  (and n should be odd)
                       otherwise the r interval is [0, r_max]
           - sigma: floats: sigma parameter for the gaussian
           - A0: float: amplitude of the gaussian
           - ratio_valid_sigma: float: in the benchmark ta
                        0 < r < ration_valid_sigma * sigma 
                        (exclude possible artefacts on the axis, and )

        Source: http://mathworld.wolfram.com/AbelTransform.html
        """

        super(GaussianAnalytical, self).__init__(n, r_max, symmetric)

        self.sigma = sigma
        self.A0 = A0

        r = self.r

        self.func = A0*np.exp(-r**2/sigma**2) 

        self.abel = sigma * np.sqrt(np.pi) * A0*np.exp(-r**2/sigma**2)

        # exclude the region near the discontinuity
        self.mask_valid = (np.abs(self.r) < ratio_valid_sigma*sigma) &\
                          (np.abs(self.r) > 0)


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
