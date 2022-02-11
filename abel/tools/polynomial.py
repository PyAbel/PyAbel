# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.linalg import pascal, toeplitz

__doc__ = """
See :ref:`Polynomials` for details and examples.

.. toctree::
    :hidden:

    tools/polynomial
"""


class BasePolynomial(object):
    """
    Abstract base class for polynomials. Implements multiplication and division
    by numbers. (Addition and subtraction of polynomials are not implemented
    because they are meaningful only for polynomials generated on the same
    grid. Use ``Piecewise...`` classes for sums of polynomials.)

    Attributes
    ----------
    func : numpy array
        values of the original function
    abel : numpy array
        values of the Abel transform
    """
    def copy(self):
        """
        Return an independent copy.
        """
        other = self.__new__(type(self))  # create empty object (same type)
        other.func = self.func.copy()
        other.abel = self.abel.copy()
        return other

    def __imul__(self, num):
        """
        In-place multiplication: Polynomial *= num.
        """
        self.func *= num
        self.abel *= num
        return self

    def __mul__(self, num):
        """
        Multiplication: Polynomial * num.
        """
        res = self.copy()
        return res.__imul__(num)

    __rmul__ = __mul__
    __rmul__.__doc__ = \
        """
        Multiplication: num * Polynomial.
        """

    def __itruediv__(self, num):
        """
        In-place division: Polynomial /= num.
        """
        return self.__imul__(1 / num)

    def __truediv__(self, num):
        """
        Division: Polynomial / num.
        """
        return self.__mul__(1 / num)


class Polynomial(BasePolynomial):
    r"""
    Polynomial function and its Abel transform.

    Parameters
    ----------
    r : numpy array
        *r* values at which the function is generated
        (and *x* values for its Abel transform);
        must be non-negative and in ascending order
    r_min, r_max : float
        *r* domain:
        the function is defined as the polynomial on [**r_min**, **r_max**]
        and zero outside it;
        0 ≤ **r_min** < **r_max** ≲ **max r**
        (**r_max** might exceed maximal **r**, but usually by < 1 pixel;
        negative **r_min** or **r_max** are allowed for convenience but are
        interpreted as 0)
    c: numpy array
        polynomial coefficients in order of increasing degree:
        [c₀, c₁, c₂] means c₀ + c₁ *r* + c₂ *r*\ ²
    r_0 : float, optional
        origin shift: the polynomial is defined as
        c₀ + c₁ (*r* − **r_0**) + c₂ (*r* − **r_0**)² + ...
    s : float, optional
        *r* stretching factor (around **r_0**): the polynomial is defined as
        c₀ + c₁ ((*r* − **r_0**)/**s**) + c₂ ((*r* − **r_0**)/**s**)² + ...
    reduced : boolean, optional
        internally rescale the *r* range to [0, 1];
        useful to avoid floating-point overflows for high degrees
        at large r (and might improve numeric accuracy)
    """
    def __init__(self, r, r_min, r_max, c, r_0=0.0, s=1.0, reduced=False):
        n = r.shape[0]

        # trim negative r limits
        if r_max <= 0:
            # both func and abel must be zero everywhere
            self.func = np.zeros(n)
            self.abel = np.zeros(n)
            return
        if r_min < 0:
            r_min = 0

        # remove zero high-order terms
        c = np.array(np.trim_zeros(c, 'b'), float)
        # if all coefficients are zero
        if len(c) == 0:
            # then both func and abel are also zero everywhere
            self.func = np.zeros(n)
            self.abel = np.zeros(n)
            return
        # polynomial degree
        K = len(c) - 1

        if reduced:
            # rescale r to [0, 1] (to avoid FP overflow)
            r = r / r_max
            r_0 /= r_max
            s /= r_max
            abel_scale = r_max
            r_min /= r_max
            r_max = 1.0

        if s != 1.0:
            # apply stretch
            S = np.cumprod([1.0] + [1.0 / s] * K)  # powers of 1/s
            c *= S
        if r_0 != 0.0:
            # apply shift
            P = pascal(1 + K, 'upper', False)  # binomial coefficients
            rk = np.cumprod([1.0] + [-float(r_0)] * K)  # powers of -r_0
            T = toeplitz([1.0] + [0.0] * K, rk)  # upper-diag. (-r_0)^{l - k}
            c = (P * T).dot(c)

        # whether even and odd powers are present
        even = np.any(c[::2])
        odd = np.any(c[1::2])

        # index limits
        i_min = np.searchsorted(r, r_min)
        i_max = np.searchsorted(r, r_max)

        # Calculate all necessary variables within [0, r_max]

        # x, x^2
        x = r[:i_max]
        x2 = x * x

        # integration limits y = sqrt(r^2 - x^2) or 0
        def sqrt0(x): return np.sqrt(x, np.zeros_like(x), where=x > 0)
        y_up = sqrt0(r_max * r_max - x2)
        y_lo = sqrt0(r_min * r_min - x2)

        # y r^k |_lo^up
        # (actually only even are neded for "even", and only odd for "odd")
        Dyr = np.outer(np.cumprod([1.0] + [r_max] * K), y_up) - \
              np.outer(np.cumprod([1.0] + [r_min] * K), y_lo)

        # ln(r + y) |_lo^up, only for odd k
        if odd:
            # ln x for x > 0, otherwise 0
            def ln0(x): return np.log(x, np.zeros_like(x), where=x > 0)
            Dlnry = ln0(r_max + y_up) - \
                    ln0(np.maximum(r_min, x) + y_lo)

        # One-sided Abel integral \int_lo^up r^k dy.
        def a(k):
            odd_k = k % 2
            # max. x power
            K = k - odd_k  # (k - 1 for odd k)
            # generate coefficients for all x^m r^{k - m} terms
            # (only even indices are actually used;
            #  for odd k, C[K] is also used for x^{k+1} ln(r + y))
            C = [0] * (K + 1)
            C[0] = 1 / (k + 1)
            for m in range(k, 1, -2):
                C[k - m + 2] = C[k - m] * m / (m - 1)
            # sum all terms using Horner's method in x
            a = C[K] * Dyr[k - K]
            if odd_k:
                a += C[K] * x2 * Dlnry
            for m in range(K - 2, -1, -2):
                a = a * x2 + C[m] * Dyr[k - m]
            return a

        # Generate the polynomial function
        func = np.zeros(n)
        span = slice(i_min, i_max)
        # (using Horner's method)
        func[span] = c[K]
        for k in range(K - 1, -1, -1):
            func[span] = func[span] * x[span] + c[k]
        self.func = func

        # Generate its Abel transform
        abel = np.zeros(n)
        span = slice(0, i_max)
        if reduced:
            c *= abel_scale
        for k in range(K + 1):
            if c[k]:
                abel[span] += c[k] * 2 * a(k)
        self.abel = abel


class PiecewisePolynomial(BasePolynomial):
    r"""
    Piecewise polynomial function (sum of :class:`Polynomial`\ s)
    and its Abel transform.

    Parameters
    ----------
    r : numpy array
        *r* values at which the function is generated
        (and *x* values for its Abel transform)
    ranges : iterable of unpackable
        (list of tuples of) polynomial parameters for each piece::

           [(r_min_1st, r_max_1st, c_1st),
            (r_min_2nd, r_max_2nd, c_2nd),
            ...
            (r_min_nth, r_max_nth, c_nth)]

        according to ``Polynomial`` conventions.
        All ranges are independent (may overlap and have gaps, may define
        polynomials of any degrees) and may include optional ``Polynomial``
        parameters

    Attributes
    ----------
    p : list of Polynomial
        :class:`Polynomial` objects corresponding to each piece
    """
    def __init__(self, r, ranges):
        self.p = [Polynomial(r, *rng) for rng in ranges]
        self.func = sum(p.func for p in self.p)
        self.abel = sum(p.abel for p in self.p)

    def copy(self):
        """
        Make an independent copy.
        """
        # make a basic copy with func and abel
        other = super(type(self), self).copy()
        # copy pieces also
        other.p = [pn.copy() for pn in self.p]
        return other

    def __imul__(self, num):
        """
        In-place multiplication: Polynomial *= num.
        """
        # multiply func and abel
        super(type(self), self).__imul__(num)
        # multiply each piece also
        for p in self.p:
            p *= num
        return self


class ApproxGaussian(object):
    r"""
    Piecewise quadratic approximation (non-negative and continuous but not
    exactly smooth) of the unit-amplitude, unit-SD Gaussian function
    :math:`\exp(-r^2/2)`, equal to it at endpoints and midpoint of each piece.
    The forward Abel transform of this approximation will typically have a
    better relative accuracy than the approximation itself.

    Parameters
    ----------
    tol : float
        absolute approximation tolerance (maximal deviation).
        Some reference values yielding the best accuracy for certain number of
        segments:

        .. table::
            :widths: auto

            =======  ===========  ===========
            **tol**  Better than  Segments
            =======  ===========  ===========
            3.7e-2   5%           3
            1.4e-2   2%           5
            4.8e-3   0.5%         7 (default)
            0.86e-3  0.1%         13
            0.99e-4  0.01%        27
            0.95e-5  0.001%       59
            =======  ===========  ===========

    Attributes
    ----------
    ranges : lists of tuple
        list of parameters ``(r_min, r_max, [c₀, c₁, c₂], r_0, s)`` that can be
        passed directly to :class:`PiecewisePolynomial` or, after
        “multiplication” by :class:`Angular`, to :class:`PiecewiseSPolynomial`
    norm : float
        the integral :math:`\int_{-\infty}^{+\infty} f(r)\,dr` for
        normalization (equals :math:`\sqrt{2\pi}` for the ideal Gaussian
        function, but slightly differs for the approximation)
    """
    def __init__(self, tol=4.8e-3):
        # Reference Gaussian function.
        def g(x):
            return np.exp(-x**2 / 2)

        # Determine splitting nodes
        xs = []  # node positions
        # first (max. x) point: g(x) = tol / 2
        x1 = np.sqrt(-2 * np.log(tol / 2))
        # moving towards x = 0...
        while x1 > 0:
            xs.append(x1)

            # Find next point x2 such that max. deviation on [x2, x1] is <= tol
            # (SciPy tools don't like this problem, so solving it manually...)

            # 3rd derivative to estimate max. deviation
            derx1 = np.abs(3 - x1**2) * x1 * g(x1)  # at x1
            # constant factor for max. of cubic Taylor term
            M = 1 / (72 * np.sqrt(3))

            # max. among mid- and endpoints
            def der(x2):
                xc = (x1 + x2) / 2
                return M * max(derx1,
                               np.abs(3 - xc**2) * xc * g(xc),
                               np.abs(3 - x2**2) * x2 * g(x2))

            # estimator of max. deviation
            def dev(x2):
                return der(x2) * (x1 - x2)**3

            x2low, x2 = x1, x1  # initialize root interval
            devx2 = 0
            for i in range(100):  # (for safety; in fact, takes ≲20 iterations)
                if devx2 > tol:  # switch to binary search (more stable)
                    xc = (x2low + x2) / 2
                    if dev(xc) > tol:
                        x2 = xc
                    else:
                        x2low = xc
                else:
                    x2low = x2
                    # estimate (x2 - x1) from ~cubic deviation growth
                    dx = (tol / der(x2))**(1/3)
                    if x2 == x1:  # for 1st estimate:
                        dx /= 2  # carefully go only 1/2 as far
                    x2 = max(x1 - dx, 0)  # shouldn't go beyond 0
                    devx2 = dev(x2)
                # terminate on root uncertainty (tol is more than enough)
                if x2low - x2 < tol:
                    break

            # make sure that outer parabola doesn't go below 0
            if len(xs) == 1 and g(x2) > 4 * g((x1 + x2) / 2):
                # use node point that matches the limiting parabola (x - x1)^2
                x2 = (x1 + 2 * np.sqrt(x1**2 - 6 * np.log(4))) / 3

            # move to next segment
            x1 = x2

        # add x = 0 to split central (last) segment if its max. deviation is
        # too large (estimated from quartic term)
        zero = (1 + g(xs[-1])) / 2 - g(xs[-1] / np.sqrt(2)) > tol

        # symmetric copy to negative x
        if zero:
            xs = [-x for x in xs] + [0.0] + xs[::-1]
        else:
            xs = [-x for x in xs] + xs[::-1]
        N = len(xs)  # total number of nodes
        xs = np.array(xs)

        # midpoints positions
        xc = (xs[:-1] + xs[1:]) / 2
        # values at nodes and midpoints
        ys = g(xs)
        ys[0] = ys[-1] = 0  # zero at endpoints
        yc = g(xc)

        # Create polynomial parameters
        cs = [ys[:-1],
              -3 * ys[:-1] + 4 * yc - ys[1:],
              2 * (ys[:-1] - 2 * yc + ys[1:])]
        self.ranges = []
        for i in range(N - 1):
            self.ranges.append((xs[i], xs[i + 1],  # r_min, r_max
                                [cs[0][i], cs[1][i], cs[2][i]],  # c
                                xs[i], xs[i + 1] - xs[i]))  # r_0, s
        # calculate norm
        self.norm = ((cs[0] + cs[1] / 2 + cs[2] / 3) * np.diff(xs)).sum()

    def scaled(self, A=1.0, r_0=0.0, sigma=1.0):
        r"""
        Parameters for piecewise polynomials corresponding to the shifted and
        scaled Gaussian function
        :math:`A \exp\big([(r - r_0)/\sigma]^2 / 2\big)`.

        (Useful numbers: a Gaussian normalized to unit integral, that is the
        standard normal distribution, has :math:`A = 1/\sqrt{2\pi}`; however,
        see :attr:`norm` above. A Gaussian with FWHM = 1 has :math:`\sigma =
        1/\sqrt{8\ln2}`.)

        Parameters
        ----------
        A : float
            amplitude
        r_0 : float
            peak position
        sigma : float
            standard deviation

        Returns
        -------
        ranges : list of tuple
            parameters for the piecewise polynomial approximating the shifted
            and scaled Gaussian
        """
        ranges = []
        for r_min, r_max, c, r, s in self.ranges:
            r_max = r_0 + sigma * r_max
            if r_max < 0:
                continue
            r_min = max(0, r_0 + sigma * r_min)
            c = [A * cn for cn in c]
            r = r_0 + sigma * r
            s *= sigma
            ranges.append((r_min, r_max, c, r, s))
        return ranges
