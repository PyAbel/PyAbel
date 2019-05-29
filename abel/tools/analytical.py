# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import
import numpy as np
import abel
import scipy.constants as const
import scipy.interpolate

# This file includes functions that have a known analytical Abel transform.
# They are used in unit testing and for comparing different Abel
# impementations.


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
        -------a--
        n : int
            number of points along the r axis

        r_max: float
            maximum r interval

        symmetric: boolean
            if True, the r interval is [-r_max, r_max] (and n should be odd),
            otherwise, the r interval is [0, r_max]
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
    """
    Define a symmetric step function and calculate its analytical
    Abel transform. See examples/example_step.py

    Parameters
    ----------
    n : int
        number of points along the r axis
    r_max : float
        range of the r interval
    symmetric : boolean
        if True, the r interval is [-r_max, r_max] (and n should be odd),
        otherwise the r interval is [0, r_max]
    r1, r2 : float
        bounds of the step function if r > 0
        (symmetric function is constructed for r < 0)
    A0: float
        height of the step
    ratio_valid_step: float
        in the benchmark take only the central ratio*100% of the step
        (exclude possible artefacts on the edges)
    """
    # see https://github.com/PyAbel/PyAbel/pull/16

    def __init__(self, n, r_max, r1, r2, A0=1.0,
                 ratio_valid_step=1.0, symmetric=True):
        super(StepAnalytical, self).__init__(n, r_max, symmetric)

        self.r1, self.r2 = r1, r2
        self.A0 = A0

        mask = np.abs(np.abs(self.r) - 0.5*(r1 + r2)) < 0.5*(r2 - r1)

        fr = np.zeros(self.r.shape)
        fr[mask] = A0

        self.func = fr

        self.abel = self.sym_abel_step_1d(self.r, self.A0, self.r1, self.r2)[0]

        # exclude the region near the discontinuity
        self.mask_valid = np.abs(
            np.abs(self.r) - 0.5*(r1 + r2)) < ratio_valid_step*0.5*(r2 - r1)

    def abel_step_analytical(self, r, A0, r0, r1):
        """
        Forward Abel transform of a step function located between r0 and r1,
        with a height A0

        ::

            A0 +                  +-------------+
               |                  |             |
               |                  |             |
             0 | -----------------+             +-------------
               +------------------+-------------+------------>
               0                  r0            r1           r axis


        Parameters
        ----------
        r1 : 1D array
            vecor of positions along the r axis. Must start with 0.
        r0, r1 : float
            positions of the step along the r axis
        A0 : float or 1D array
            height of the step. If 1D array, the height can be variable
            along the z axis

        Returns
        -------
            1D array, if A0 is a float, a 2D array otherwise
        """

        if np.all(r[[0, -1]]):
            raise ValueError('The vector of r coordinates must start with 0.0')

        F_1d = np.zeros(r.shape)
        mask = (r >= r0)*(r < r1)
        F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2)

        mask = r < r0
        F_1d[mask] = 2*np.sqrt(r1**2 - r[mask]**2) -\
                     2*np.sqrt(r0**2 - r[mask]**2)

        A0 = np.atleast_1d(A0)
        if A0.ndim == 1:
            A0 = A0[:, np.newaxis]

        return F_1d[np.newaxis, :]*A0

    def sym_abel_step_1d(self, r, A0, r0, r1):
        """
        Produces a symmetrical analytical transform of a 1D step
        """
        A0 = np.atleast_1d(A0)
        d = np.empty(A0.shape + r.shape)
        A0 = A0[np.newaxis, :]

        for sens, mask in enumerate([r >= 0, r <= 0]):
            d[:, mask] = self.abel_step_analytical(np.abs(r[mask]), A0, r0, r1)

        return d


class Polynomial(BaseAnalytical):
    """
    Define a polynomial function and calculate its analytical
    Abel transform.

    (See :ref:`Polynomials` for details and examples.)

    Parameters
    ----------
    n : int
        number of points along the *r* axis
    r_max : float
        range of the *r* interval
    symmetric : boolean
        if ``True``, the *r* interval is [−\ **r_max**, +\ **r_max**]
        (and **n** should be odd),
        otherwise the *r* interval is [0, **r_max**]
    r_1, r_2 : float
        *r* bounds of the polynomial function if *r* > 0;
        outside [**r_1**, **r_2**] the function is set to zero
        (symmetric function is constructed for *r* < 0)
    c: numpy array
        polynomial coefficients in order of increasing degree:
        [c₀, c₁, c₂] means c₀ + c₁ *r* + c₂ *r*\ ²
    r_0 : float, optional
        origin shift: the polynomial is defined as
        c₀ + c₁ (*r* − **r_0**) + c₂ (*r* − **r_0**)² + ...
    s : float, optional
        *r* stretching factor (around **r_0**): the polynomial is defined as
        c₀ + c₁ (*r*/s) + c₂ (*r*/s)² + ...
    reduced : boolean, optional
        internally rescale the *r* range to [0, 1];
        useful to avoid floating-point overflows for high degrees
        at large *r* (and might improve numerical accuracy)
    """
    def __init__(self, n, r_max,
                 r_1, r_2, c, r_0=0.0, s=1.0, reduced=False,
                 symmetric=True):
        super(Polynomial, self).__init__(n, r_max, symmetric)

        # take r >= 0 part
        if symmetric:
            r = self.r[n//2:]
        else:
            r = self.r

        P = abel.tools.polynomial.Polynomial(r, r_1, r_2, c, r_0, s, reduced)
        self.func = P.func
        self.abel = P.abel

        # mirror to negative r, if needed
        if symmetric:
            self.func = np.hstack((self.func[:0:-1], self.func))
            self.abel = np.hstack((self.abel[:0:-1], self.abel))

        self.mask_valid = np.ones_like(self.func)


class PiecewisePolynomial(BaseAnalytical):
    """
    Define a piecewise polynomial function (sum of ``Polynomial``\ s)
    and calculate its analytical Abel transform.

    Parameters
    ----------
    n : int
        number of points along the *r* axis
    r_max : float
        range of the *r* interval
    symmetric : boolean
        if ``True``, the *r* interval is [−\ **r_max**, +\ **r_max**]
        (and **n** should be odd),
        otherwise the *r* interval is [0, **r_max**]
    ranges : iterable of unpackable
        (list of tuples of) polynomial parameters for each piece::

           [(r_1_1st, r_2_1st, c_1st),
            (r_1_2nd, r_2_2nd, c_2nd),
            ...
            (r_1_nth, r_2_nth, c_nth)]

        according to ``Polynomial`` conventions.
        All ranges are independent (may overlap and have gaps, may define
        polynomials of any degrees) and may include optional ``Polynomial``
        parameters
    """
    def __init__(self, n, r_max,
                 ranges,
                 symmetric=True):
        super(PiecewisePolynomial, self).__init__(n, r_max, symmetric)

        # take r >= 0 part
        if symmetric:
            r = self.r[n//2:]
        else:
            r = self.r

        P = abel.tools.polynomial.PiecewisePolynomial(r, ranges)
        self.func = P.func
        self.abel = P.abel

        # mirror to negative r, if needed
        if symmetric:
            self.func = np.hstack((self.func[:0:-1], self.func))
            self.abel = np.hstack((self.abel[:0:-1], self.abel))

        self.mask_valid = np.ones_like(self.func)


class GaussianAnalytical(BaseAnalytical):
    """
    Define a gaussian function and calculate its analytical
    Abel transform. See examples/example_gaussian.py

    Parameters
    ----------
    n : int
        number of points along the r axis
    r_max : float
        range of the r interval
    symmetric : boolean
        if True, the r interval is [-r_max, r_max] (and n should be odd),
        otherwise, the r interval is [0, r_max]
    sigma : floats
        sigma parameter for the gaussian
    A0 : float
        amplitude of the gaussian
    ratio_valid_sigma : float
        in the benchmark take only the range
        0 < r < ration_valid_sigma * sigma
        (exclude possible artefacts on the axis and the possibly clipped tail)
    """
    # Source: http://mathworld.wolfram.com/AbelTransform.html

    def __init__(self, n, r_max, sigma=1.0, A0=1.0,
                 ratio_valid_sigma=2.0, symmetric=True):
        super(GaussianAnalytical, self).__init__(n, r_max, symmetric)

        self.sigma = sigma
        self.A0 = A0

        r = self.r

        self.func = A0*np.exp(-r**2/sigma**2)

        self.abel = sigma * np.sqrt(np.pi) * A0*np.exp(-r**2/sigma**2)

        # exclude the region near the discontinuity
        self.mask_valid = (np.abs(self.r) < ratio_valid_sigma*sigma) &\
                          (np.abs(self.r) > 0)


class TransformPair(BaseAnalytical):
    """**Abel-transform pair analytical functions**.

    **profiles 1–7**: Table 1 of
    `Chan and Hieftje Spectrochimica Acta B 61, 31–41 (2006)
    <http://doi.org/10.1016/j.sab.2005.11.009>`_.

    See :mod:`abel.tools.transform_pairs`.


    Returns
    -------
    r : numpy array
        vector of positions along the r axis: `linspace(0, 1, n)`

    dr : float
        radial interval

    func : numpy array
        values of the original function (same shape as r)

    abel : numpy array
        values of the Abel transform (same shape as func)

    label : str
        name of the curve

    mask_valid : boolean array
        set all True. Used for unit tests

    """

    def __init__(self, n, profile=5):
        """Create Abel transform pair for given profile number.

        Parameters
        ----------
        n : int
            number of points along the r axis

        profile: int
            the profile number 1-7, see 'abel/tools/transform_pairs.py'

        """

        super(TransformPair, self).__init__(n, r_max=1, symmetric=False)

        # BaseAnalytical creates self.r = linspace(0, 1, n)
        # prevent divide by zero and NaN for r = 0, or 1,
        # slightly offset these values
        r = self.r.copy()
        r[0] = 1.0e-8
        r[-1] -= 1.0e-8

        if profile > 7:
            raise ValueError('Only 1-7 profiles: '
                             'see "abel/tools/transform_pairs.py"')

        self.label = 'profile{}'.format(profile)

        self.profile = getattr(abel.tools.transform_pairs, self.label)
        self.func, self.abel = self.profile(r)

        # function values to use for testing
        self.mask_valid = np.ones_like(self.func)


class SampleImage(BaseAnalytical):
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

    Attributes
    ----------
    image : 2D numpy array
         image

    name : str
         sample image name
    """
    def __init__(self, n=361, name="dribinski", sigma=2, temperature=200):

        def _gauss(r, r0, sigma):
            return np.exp(-(r-r0)**2/sigma**2)

        def _dribinski(r, theta, sigma):
            """
            Sample test image used in the BASEX paper
            Rev. Sci. Instrum. 73, 2634 (2002), intensity function Eq. (16)
            (there are some missing negative exponents in the publication)

            9 Gaussian peaks with "width" = sigma (default: 2) and
            1 background Gaussian with "width" = 60
            ("width" is √2 std. dev. in pixels for default n = 361
             and scales proportionally to n)

            anisotropy: ß = −1  for cosθ term
                        ß = +2  for sinθ term
                        ß =  0  isotropic, no angular variation
            """

            sinetheta2 = np.sin(theta)**2
            cosinetheta2 = np.cos(theta)**2

            t0 = 7*_gauss(r, 10, sigma)*sinetheta2
            t1 = 3*_gauss(r, 15, sigma)
            t2 = 5*_gauss(r, 20, sigma)*cosinetheta2
            t3 = _gauss(r, 70, sigma)
            t4 = 2*_gauss(r, 85, sigma)*cosinetheta2
            t5 = _gauss(r, 100, sigma)*sinetheta2
            t6 = 2*_gauss(r, 145, sigma)*sinetheta2
            t7 = _gauss(r, 150, sigma)
            t8 = 3*_gauss(r, 155, sigma)*cosinetheta2
            t9 = 20*_gauss(r, 45, 60)  # background under t3 to t5

            return 2000*(t0+t1+t2) + 200*(t3+t4+t5) + 50*(t6+t7+t8) + t9

        def _Ominus(r, theta, sigma, boltzmann, rfact=1):
            """
            Simulate the photoelectron spectrum of O- photodetachment
            3PJ <- 2P3/2,1/2

            6 transitions, triplet neutral, and doublet anion

            """
            # positions based on 812.5 nm ANU O- PES
            t1 = _gauss(r, 341*rfact, sigma)  # 3P2 <- 2P3/2
            t2 = _gauss(r, 285*rfact, sigma)  # 3P1 <- 2P3/2
            t3 = _gauss(r, 257*rfact, sigma)  # 3P0 <- 2P3/2
            t4 = _gauss(r, 394*rfact, sigma)  # 3P2 <- 2P1/2
            t5 = _gauss(r, 348*rfact, sigma)  # 3P1 <- 2P1/2
            t6 = _gauss(r, 324*rfact, sigma)  # 3P0 <- 2P1/2

            # intensities = fine-structure known ratios
            # 2P1/2 transtions scaled by temperature-dependent Boltzmann factor
            t = t1 + 0.8*t2 + 0.36*t3 + (0.2*t4 + 0.36*t5 + 0.16*t6)*boltzmann

            # anisotropy
            sinetheta2 = np.sin(theta)**2*0.2 + 0.8

            return t*sinetheta2

        n2 = n//2
        self.name = name

        super(SampleImage, self).__init__(n, r_max=n2, symmetric=True)

        if name == 'dribinski':
            scale = 180*2/n
        elif name == 'Ominus':
            scale = 501*2/n

        X, Y = np.meshgrid(self.r*scale, self.r*scale)
        R, THETA = abel.tools.polar.cart2polar(X, Y)

        if self.name == "dribinski":
            self.image = _dribinski(R, THETA, sigma=sigma)
        elif self.name == "Ominus":
            boltzmann = np.exp(-177.1*const.h*const.c*100/const.k/temperature)\
                        / 2
            self.image = _Ominus(R, THETA, sigma=sigma, boltzmann=boltzmann)
        else:
            raise ValueError('Sample image name not recognized')
