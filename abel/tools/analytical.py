# -*- coding: utf-8 -*-
import numpy as np
from abel.tools.polar import index_coords, cart2polar
import scipy.constants as const

# This file includes functions that have a known analytical Abel transform.
# They are used in unit testing as well for comparing different Abel
# impementations.


def sample_image(n=361, name="dribinski", sigma=2, temperature=200):
    """
    Sample images, made up of Gaussian functions

    Parameters
    ----------
    n : integer
        image size n rows x n cols

    name : str
        one of "dribinski" or "Ominus"

    sigma : float
        Gaussian 1/e width (pixels)

    temperature : float
        for 'Ominus' only
        anion levels have Boltzmann population weight
        (2J+1) exp(-177.1 h c 100/k/temperature)

    Returns
    -------
    IM : 2D np.array
        image
    """

    def gauss(r, r0, sigma):
        return np.exp(-(r-r0)**2/sigma**2)

    def dribinski(r, theta, sigma):  # intensity function Eq. (16)
        """
        Sample test image used in the BASEX paper
        Rev. Sci. Instrum. 73, 2634 (2002)

        9x Gaussian functions of half-width 4 pixel + 1 background width 3600
        anisotropy - ß = -1  for cosθ term
                   - ß = +2  for sinθ term
                   - ß =  0  isotropic, no angular variation
        (there are some missing negative exponents in the publication)
        """

        sinetheta2 = np.sin(theta)**2
        cosinetheta2 = np.cos(theta)**2

        t0 = 7*gauss(r, 10, sigma)*sinetheta2
        t1 = 3*gauss(r, 15, sigma)
        t2 = 5*gauss(r, 20, sigma)*cosinetheta2
        t3 = gauss(r, 70, sigma)
        t4 = 2*gauss(r, 85, sigma)*cosinetheta2
        t5 = gauss(r, 100, sigma)*sinetheta2
        t6 = 2*gauss(r, 145, sigma)*sinetheta2
        t7 = gauss(r, 150, sigma)
        t8 = 3*gauss(r, 155, sigma)*cosinetheta2
        t9 = 20*gauss(r, 45, 3600)  # background under t3 to t5
        return 2000*(t0+t1+t2) + 200*(t3+t4+t5) + 50*(t6+t7+t8) + t9

    def Ominus(r, theta, sigma, boltzmann, rfact=1):
        """
        Simulate the photoelectron spectrum of O- photodetachment
        3PJ <- 2P3/2,1/2

        6 transitions, triplet neutral, and doublet anion

        """
        # positions based on 812.5 nm ANU O- PES
        t1 = gauss(r, 341*rfact, sigma)  # 3P2 <- 2P3/2
        t2 = gauss(r, 285*rfact, sigma)  # 3P1 <- 2P3/2
        t3 = gauss(r, 257*rfact, sigma)  # 3P0 <- 2P3/2
        t4 = gauss(r, 394*rfact, sigma)  # 3P2 <- 2P1/2
        t5 = gauss(r, 348*rfact, sigma)  # 3P1 <- 2P1/2
        t6 = gauss(r, 324*rfact, sigma)  # 3P0 <- 2P1/2

        # intensities = fine-structure known ratios
        # 2P1/2 transtions scaled by temperature dependent Boltzmann factor
        t = t1 + 0.8*t2 + 0.36*t3 + (0.2*t4 + 0.36*t5 + 0.16*t6)*boltzmann

        # anisotropy
        sinetheta2 = np.sin(theta)**2*0.2 + 0.8

        return t*sinetheta2

    n2 = n//2

    
    x = np.linspace(-n2, n2, n)
    
    if name=='dribinski': 
        x = x * 180/n2
    elif name=='Ominus':
        x = x * 501/n2
    

    X, Y = np.meshgrid(x, x)
    R, THETA = cart2polar(X, Y)
    if name == "dribinski":
        IM = dribinski(R, THETA, sigma=sigma)
    elif name == "Ominus":
        boltzmann = 0.5*np.exp(-177.1*const.h*const.c*100/const.k/temperature)
        IM = Ominus(R, THETA, sigma=sigma, boltzmann=boltzmann)
    else:
        raise ValueError('sample image name not recognized')

    return IM


class BaseAnalytical(object):

    def __init__(self, n, r_max, symmetric=True, **args):
        """
        This is the base class for functions that have a known Abel transform.
        Every such class should expose the following public attributes:
          - self.r: vector of positions along the r axis
          - self.func: the values of the original function
                        (same shape as self.r)
          - self.abel: the values of the Abel transform (same shape as self.r)
          - self.mask_valid: mask of the r interval where the function is
            well smoothed/well behaved (no known artefacts in the inverse
            Abel reconstuction), typically excluding the origin, the domain
            boundaries, and function discontinuities, that can be used for
            unit testing.

        See GaussianAnalytical for a concrete example.

        Parameters
        ----------
        n : int
            number of points along the r axis

        r_max: float
            maximum r interval

        symmetric: boolean
            if True the r interval is [-r_max, r_max]  (and n should be odd)
                          otherwise the r interval is [0, r_max]
        """
        self.n = n
        self.r_max = r_max

        assert r_max > 0

        if symmetric:
            self.r = np.linspace(-r_max, r_max, n)
        else:
            self.r = np.linspace(0, r_max, n)

        self.dr = np.diff(self.r)[0]


class StepAnalytical(BaseAnalytical):
    def __init__(self, n, r_max, r1, r2, A0=1.0,
                 ratio_valid_step=1.0, symmetric=True):
        """
        Define a a symmetric step function and calculate it's analytical
        Abel transform. See examples/example_step.py

        Parameters
        ----------
        n : int
            number of points along the r axis
        r_max: float
            range of the symmetric r interval
        symmetric: if True
            the r interval is [-r_max, r_max]
            (and n should be odd)
            otherwise the r interval is [0, r_max]
        r1, r2: floats
            bounds of the step function if r > 0
            (symetic function is constructed for r < 0)
        A0: float
            height of the step
        ratio_valid_step: float
            in the benchmark take only the central ratio*100% of the step
            (exclude possible artefacts on the edges)

        see https://github.com/PyAbel/PyAbel/pull/16

        """

        super(StepAnalytical, self).__init__(n, r_max, symmetric)

        self.r1, self.r2 = r1, r2
        self.A0 = A0

        mask = np.abs(np.abs(self.r) - 0.5*(r1 + r2)) < 0.5*(r2 - r1)

        fr = np.zeros(self.r.shape)
        fr[mask] = A0

        self.func = fr

        self.abel = sym_abel_step_1d(self.r, self.A0, self.r1, self.r2)[0]

        # exclude the region near the discontinuity
        self.mask_valid = np.abs(
            np.abs(self.r) - 0.5*(r1 + r2)) < ratio_valid_step*0.5*(r2 - r1)


class GaussianAnalytical(BaseAnalytical):
    def __init__(self, n, r_max, sigma=1.0, A0=1.0,
                 ratio_valid_sigma=2.0, symmetric=True):
        """
        Define a gaussian function and calculate its analytical
        Abel transform. See examples/example_gaussian.py

        Parameters
        n : int
            number of points along the r axis
        r_max: float
            range of the symmetric r interval
        symmetric: if True
            the r interval is [-r_max, r_max]
            (and n should be odd)
            otherwise the r interval is [0, r_max]
        sigma: floats
            sigma parameter for the gaussian
        A0: float
            amplitude of the gaussian
        ratio_valid_sigma: float
            in the benchmark ta
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
    Directed Abel transform of a step function located between r0 and r1,
    with a height A0

    ::

        A0 +                  +-------------+
           |                  |             |
           |                  |             |
         0 | -----------------+             +-------------
           +------------------+-------------+------------>
           0                  r0            r1           r axis

    This function is mostly used for unit testing the inverse Abel transform

    Parameters
    ----------
    r1 : 1D array,
        vecor of positions along the r axis. Must start with 0.
    r0, r1 : floats,
        positions of the step along the r axis
    A0 : float or 1D array:
        height of the step. If 1D array the height can be variable
        along the Z axis

    Returns
    -------
        1D array if A0 is a float, a 2D array otherwise
    """

    if np.all(r[[0, -1]]):
        raise ValueError('The vector of r coordinates must start with 0.0')

    F_1d = np.zeros(r.shape)
    mask = (r >= r0)*(r < r1)
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2)
    mask = r < r0
    F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2) - 2*np.sqrt(r0**2 - r[mask]**2)
    A0 = np.atleast_1d(A0)
    if A0.ndim == 1:
        A0 = A0[:, np.newaxis]
    return F_1d[np.newaxis, :]*A0


def sym_abel_step_1d(r, A0, r0, r1):
    """
    Produces a symmetrical analytical transform of a 1d step
    """
    A0 = np.atleast_1d(A0)
    d = np.empty(A0.shape + r.shape)
    A0 = A0[np.newaxis, :]
    for sens, mask in enumerate([r >= 0, r <= 0]):
        d[:, mask] = abel_step_analytical(np.abs(r[mask]), A0, r0, r1)

    return d
