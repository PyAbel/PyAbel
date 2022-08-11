# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import absolute_import
import numpy as np
import scipy.constants as const
import scipy.interpolate

import abel
from abel import _deprecate

# This file includes functions that have a known analytical Abel transform.
# They are used in unit testing and for comparing different Abel
# impementations.


class BaseAnalytical(object):
    r"""
    Base class for functions that have a known Abel transform
    (see :class:`GaussianAnalytical` for a concrete example).
    Every such class should expose the following public attributes:

    Attributes
    ----------
    r : numpy array
        vector of positions along the :math:`r` axis
    func : numpy array
        the values of the original function (same shape as :attr:`r` for 1D
        functions, or same row size as :attr:`r` for 2D images)
    abel : numpy array
        the values of the Abel transform (same shape as :attr:`func`)
    mask_valid : numpy array
        mask (same shape as :attr:`func`) where the function is
        well smoothed/well behaved (no known artefacts in the inverse
        Abel reconstuction), typically excluding the origin, the domain
        boundaries, and function discontinuities, that can be used for
        unit testing.

    Parameters
    ----------
    n : int
        number of points along the :math:`r` axis
        (saved to attribute :attr:`n`)
    r_max: float
        maximum :math:`r` value (saved to attribute :attr:`r_max`)
    symmetric: boolean
        if ``True``, the :math:`r` interval is [−\ **r_max**, **r_max**]
        (and **n** should be odd),
        otherwise, the :math:`r` interval is [0, **r_max**]
    """
    def __init__(self, n, r_max, symmetric=True, **args):
        self.n = n
        self.r_max = r_max

        assert r_max > 0

        if symmetric:
            self.r = np.linspace(-r_max, r_max, n)
        else:
            self.r = np.linspace(0, r_max, n)

        self.dr = np.diff(self.r)[0]


class StepAnalytical(BaseAnalytical):
    r"""
    Define a step function and calculate its analytical Abel transform:

    ..
        A0 +                  +-------------+
           |                  |             |
           |                  |             |
         0 | -----------------+             +-------------
           +------------------+-------------+------------>
           0                  r1            r2           r axis

    .. plot:: tools/step_analytical.py

    See :doc:`examples/example_basex_step.py <example_basex_step>`.

    Parameters
    ----------
    n : int
        number of points along the `r` axis
    r_max : float
        range of the `r` interval
    symmetric : boolean
        if ``True``, the `r` interval is [−\ **r_max**, **r_max**]
        (and **n** should be odd),
        otherwise the `r` interval is [0, **r_max**]
    r1, r2 : float
        bounds of the step function for `r` > 0
        (symmetric function is constructed for `r` < 0)
    A0: float
        height of the step
    ratio_valid_step: float
        in the benchmark take only the central ratio × 100% of the step
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
        with a height A0.

        Parameters
        ----------
        r : 1D array
            array of positions along the r axis. Must start with 0.
        A0 : float or 1D array
            height of the step. If 1D array, the height can be variable
            along the z axis
        r0, r1 : float
            positions of the step along the r axis

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
    r"""
    Define a polynomial function and calculate its analytical
    Abel transform.

    (See :ref:`Polynomials` for details and examples.)

    Parameters
    ----------
    n : int
        number of points along the *r* axis
    r_max : float
        range of the *r* interval
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
    symmetric : boolean
        if ``True``, the *r* interval is [−\ **r_max**, +\ **r_max**]
        (and **n** should be odd),
        otherwise the *r* interval is [0, **r_max**]
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
    r"""
    Define a piecewise polynomial function (sum of ``Polynomial``\ s)
    and calculate its analytical Abel transform.

    Parameters
    ----------
    n : int
        number of points along the *r* axis
    r_max : float
        range of the *r* interval
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
    symmetric : boolean
        if ``True``, the *r* interval is [−\ **r_max**, +\ **r_max**]
        (and **n** should be odd),
        otherwise the *r* interval is [0, **r_max**]
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
    Abel transform. See :doc:`examples/example_basex_gaussian.py
    <example_basex_gaussian>`.

    Parameters
    ----------
    n : int
        number of points along the r axis
    r_max : float
        range of the r interval
    sigma : float
        sigma parameter for the gaussian
    A0 : float
        amplitude of the gaussian
    ratio_valid_sigma : float
        in the benchmark take only the range
        0 < r < ration_valid_sigma * sigma
        (exclude possible artefacts on the axis and the possibly clipped tail)
    symmetric : boolean
        if True, the r interval is [-r_max, r_max] (and n should be odd),
        otherwise, the r interval is [0, r_max]
    """
    # Source: https://mathworld.wolfram.com/AbelTransform.html

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
    G. C.-Y. Chan, Gary M. Hieftje,
    "Estimation of confidence intervals for radial emissivity and optimization
    of data treatment techniques in Abel inversion",
    `Spectrochimica Acta B 61, 31–41 (2006)
    <https://doi.org/10.1016/j.sab.2005.11.009>`_.

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
    r"""
    Sample images, made up of Gaussian functions
    (or cubic splines, for ``'O2'``).

    Parameters
    ----------
    n : integer
        image size **n** rows × **n** cols (must be odd for most purposes;
        even **n** values would result in half-pixel centering)
    name : str
        ``'Dribinski'``
            Sample test image used in the BASEX paper `Rev. Sci. Instrum. 73,
            2634 (2002) <https://dx.doi.org/10.1063/1.1482156>`__, intensity
            function Eq. (16) (there are some missing negative exponents in the
            publication).

            9 Gaussian peaks with alternating anisotropies (β = −1, 0, +2),
            plus 1 wide background Gaussian. Peak amplitudes are designed to
            produce comparable heights in the *speed distribution*, thus the
            peaks at small radii appear very intense in the image and its Abel
            transform.
        ``'Gaussian'``
            Isotropic 2D Gaussian :math:`\exp(-r^2 / \textbf{sigma}^2)`.

            Its Abel transform is also a Gaussian with the same width:
            :math:`\sqrt{\pi}\, \textbf{sigma} \exp(-r^2 / \textbf{sigma}^2)`.
        ``'Gerber'``
            Artificial test image used in the lin-BASEX article
            `Rev. Sci. Instrum. 84, 033101 (2013)
            <https://dx.doi.org/10.1063/1.4793404>`__, Table I.

            8 Gaussian peaks with various intensities and anisotropies up to
            4th order (β\ :sub:`4`).
        ``'O2'``
            Synthetic image mimicking a velocity-map image of O\ :sup:`+` from
            multiphoton photodissociation/ionization
            :math:`{\rm O}_2 \xrightarrow{4h\nu} {\rm O} + {\rm O}^+ + e^-`
            at :math:`44\,444~\text{cm}^{-1}` (225 nm)

            Multiple peaks with various intensities and anisotropies; see, for
            example, `J. Chem. Phys. 107, 2357 (1997)
            <https://doi.org/10.1063/1.474624>`__.
        ``'Ominus'`` or ``'O-'``
            Simulate the photoelectron spectrum of O\ :sup:`−` photodetachment
            :math:`{}^3P_J \gets {}^2P_{3/2,1/2}`.

            6 transitions, triplet neutral, and doublet anion.
    sigma : float
        1/*e* halfwidth of peaks in pixels, default values are:
        2⋅\ *r*\ :sub:`max`/180 for ``'Dribinski'``,
        2⋅\ *r*\ :sub:`max`/500 for ``'Ominus'``,
        *r*\ :sub:`max`/3 for ``'Gaussian'``,
        :math:`\sqrt{2}` (std. dev. = 1) for ``'Gerber'``.

        For ``'O2'``: HWHM of narrow peaks in pixels, default is 1.5 for any
        *r*\ :sub:`max`.
    temperature : float
        anion temperature in kelvins (default: 200) for ``'Ominus'``:
        anion levels have Boltzmann population weight :math:`(2J + 1)
        \exp[-hc \cdot 177.1~\text{cm}^{-1}/(k \cdot \textbf{temperature})]`

    Attributes
    ----------
    name : str
         sample-image name
    """
    def __init__(self, n=361, name='Dribinski', sigma=None, temperature=200):
        super(SampleImage, self).__init__(n, r_max=(n - 1) / 2, symmetric=True)

        name = name.lower()
        if name == 'dribinski':
            self.name = 'Dribinski'
            self._scale = self.r_max / 180
            width = 2 * self._scale if sigma is None else sigma
            bg_width = 60 * self._scale
            cos2 = [0, 0, 1]
            sin2 = [1, 0, -1]
            iso = [1]
            # parameters:   A         r0   width     angular
            self._peaks = [(2000 * 7,  10, width,    sin2),
                           (2000 * 3,  15, width,    iso),
                           (2000 * 5,  20, width,    cos2),
                           (200 * 1,   70, width,    iso),
                           (200 * 2,   85, width,    cos2),
                           (200 * 1,  100, width,    sin2),
                           (50 * 2,   145, width,    sin2),
                           (50 * 1,   150, width,    iso),
                           (50 * 3,   155, width,    cos2),
                           (20 * 1,    45, bg_width, iso)]
        elif name == 'gaussian':
            self.name = 'Gaussian'
            self._scale = 0
            width = self.r_max / 3 if sigma is None else sigma
            # parameters:   A  r0 width  angular
            self._peaks = [(1, 0, width, [1])]
        elif name == 'gerber':
            self.name = 'Gerber'
            self._scale = self.r_max / 256
            width = np.sqrt(2) if sigma is None else sigma

            def sphere(r, beta0, beta2, beta4):
                N = 160000 / (4 * np.pi * np.sqrt(np.pi) * width)
                return (N / r**2, r, width,
                        beta0 * np.array([1, 0, 0, 0, 0]) +
                        beta2 * np.array([-1/2, 0, 3/2, 0, 0]) +
                        beta4 * np.array([3/8, 0, -30/8, 0, 35/8]))
            # parameters:         r0   beta0 beta2 beta4
            self._peaks = [sphere(38,  1.2, -0.4,  0),
                           sphere(70,  1.5, -1,    0.5),
                           sphere(90,  1.5,  1,    0.4),
                           sphere(134, 2,    1,    0.4),
                           sphere(138, 1.8,  0.5,  0),
                           sphere(143, 1,    0.5,  0.25),
                           sphere(196, 2,    1,   -0.5),
                           sphere(230, 2,    1,   -0.5)]
        elif name == 'o2':
            self.name = 'O2'
            self._scale = self.r_max / 330
            narrow = 1.5 if sigma is None else sigma
            broad = 2 * narrow
            # fitted to Fig. 3.9 from M. Ryazanov, Ph.D. dissertation (2012)
            # parameters:   A      r0   width   angular
            self._peaks = [(5,       0, broad,  [1]),
                           (0.15,   63, narrow, [1, 0, 0.5]),
                           (0.08,   69, narrow, [0, 0, 1]),
                           (0.1,    74, narrow, [0, 0, 1]),
                           (0.15,   95, narrow, [0.3, 0, 1]),
                           (0.7,   111, broad,  [0.3, 0, 1]),
                           (1.0,   141, broad,  [0.05, 0, 1]),
                           (0.5,   153, narrow, [0.3, 0, 1]),
                           (0.03,  172, narrow, [0, 0, 1]),
                           (0.06,  180, narrow, [0, 0, 1]),
                           (0.04,  190, narrow, [0.3, 0, 1]),
                           (0.02,  201, narrow, [0.3, 0, 1]),
                           (0.007, 213, narrow, [0, 0, 1]),
                           (0.007, 227, narrow, [0, 0, 1]),
                           (0.015, 239, narrow, [0.5, 0, 1]),
                           (0.01,  250, narrow, [0, 0, 1]),
                           (0.03,  261, narrow, [0.4, 0, 1]),
                           (0.05,  270, narrow, [0, 0, 1]),
                           (0.015, 278, narrow, [0, 0, 1]),
                           (0.03,  286, narrow, [0, 0, 1]),
                           (0.015, 294, narrow, [0, 0, 1]),
                           (0.03,  301, narrow, [0, 0, 1]),
                           (0.13,  307, narrow, [0, 0, 1]),
                           (0.06,  313, narrow, [0, 0, 1])]
        elif name in ['ominus', 'o-']:
            self.name = 'Ominus'
            self._scale = self.r_max / 500
            width = 2 * self._scale if sigma is None else sigma
            boltzmann = np.exp(-177.1 * const.h * const.c * 100 /
                               (const.k * temperature)) / 2
            aniso = [1, 0, -0.2]
            # parameters:   A                 r0   width  angular
            self._peaks = [(1.0,              341, width, aniso),
                           (0.8,              285, width, aniso),
                           (0.36,             257, width, aniso),
                           (boltzmann * 0.2,  394, width, aniso),
                           (boltzmann * 0.36, 348, width, aniso),
                           (boltzmann * 0.16, 324, width, aniso)]
        else:
            raise ValueError('Sample image name not recognized.')

        # Isotropic ring for one peak.
        def peak(r, r0, width):
            if self.name == 'O2':
                # cubic spline ("width" is HWHM)
                dr = np.abs(r - r0) / (2 * width)
                f = 1 - (3 - 2 * dr) * dr**2
                f[dr > 1] = 0
                return f
            else:
                # Gaussian function ("width" is √2 std. dev.)
                return np.exp(-((r - r0) / width)**2)

        # calculate only one quadrant Q1
        n2 = (n + 1) // 2
        r0 = -self.r[n2]
        R, COS = abel.tools.polynomial.rcos(shape=(n2, n2), origin=(r0, r0))
        Q = np.zeros_like(R)
        for A, r0, width, cn in self._peaks:
            ring = A * peak(R, r0 * self._scale, width)
            Q += cn[0] * ring
            for c in cn[1:]:
                ring *= COS
                if c:
                    Q += c * ring

        # unfold to whole image
        self.func = abel.tools.symmetry.put_image_quadrants(
                        [Q[:, ::-1]] * 4, (n, n))
        self.mask_valid = abel.tools.symmetry.put_image_quadrants(
                              [R[:, ::-1] != 0] * 4, (n, n))

    def transform(self, tol=4.8e-3):
        """
        Compute forward Abel transform of the image as an analytical Abel
        transform of its piecewise polynomial approximation (except
        ``'Gaussian'`` and ``'O2'``, which are computed exactly).

        Parameters
        ----------
        tol : float
            relative tolerance of the approximation (max. deviation divided by
            max. amplitude, default: 4.8e-3 ≲ 0.5%); the resulting Abel
            transform is somewhat more accurate

        Returns
        -------
        abel : 2D numpy array
            Abel-transformed image, also accessible as the :attr:`abel`
            attribute
        """
        # calculate only one quadrant Q1
        n2 = (self.n + 1) // 2
        r0 = -self.r[n2]
        R, COS = abel.tools.polynomial.rcos(shape=(n2, n2), origin=(r0, r0))
        if self.name == 'Gaussian':
            A, r0, width, cn = self._peaks[0]
            Q = np.sqrt(np.pi) * width * np.exp(-(R / width)**2)
        else:
            # Radial polynomial parameters for one peak.
            if self.name == 'O2':
                def peak(A, r0, width):
                    width *= 2
                    c = [A, 0, -3 * A, 2 * A]
                    return [(r0 - width, r0, c, r0, -width),
                            (r0, r0 + width, c, r0, width)]
            else:
                # approximate Gaussian
                g = abel.tools.polynomial.ApproxGaussian(tol)

                def peak(A, r0, width):
                    return g.scaled(A, r0, width / np.sqrt(2))

            # sum all peaks
            Q = np.zeros_like(R)
            for A, r0, width, cn in self._peaks:
                Q += abel.tools.polynomial.PiecewiseSPolynomial(
                        R, COS,
                        peak(A, r0 * self._scale, width) *
                        abel.tools.polynomial.Angular(cn)
                     ).abel

        # unfold to whole image
        self._abel = abel.tools.symmetry.put_image_quadrants(
                         [Q[:, ::-1]] * 4, (self.n, self.n))
        return self._abel

    @property
    def image(self):
        """Deprecated. Use :attr:`func` instead."""
        _deprecate('SampleImage attribute ".image" is deprecated, '
                   'use ".func" instead.')
        return self.func

    @property
    def abel(self):
        """
        Abel transform of the image, computed (with default accuracy) only if
        necessary; see :meth:`transform` for details.
        """
        try:
            return self._abel
        except AttributeError:  # not yet computed
            return self.transform()
