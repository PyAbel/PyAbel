# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from scipy.linalg import pascal, invpascal, toeplitz
from scipy.special import legendre
from scipy.interpolate import PPoly, UnivariateSpline
try:
    from functools import cache
except ImportError:  # no functools in Python 2
    def cache(func):
        res = {}

        def decorated(*args):
            if args not in res:
                res[args] = func(*args)
            return res[args]

        return decorated

__doc__ = """
See :ref:`Polynomials` for details and examples.
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


class SPolynomial(BasePolynomial):
    r"""
    Bivariate polynomial function :math:`\sum_{mn} c_{mn} r^m \cos^n\theta` in
    spherical coordinates and its Abel transform.

    Parameters
    ----------
    r, cos : numpy array
        *r* and cos *θ* values at which the function is generated; *r* must be
        non-negative. Arrays for generating a 2D image can be conveniently
        prepared by the :func:`rcos` function. On the other hand, the radial
        dependence alone (for a *single* cosine power) can be obtained by
        supplying a 1D **r** array and a **cos** array filled with ones.
    r_min, r_max : float
        *r* domain: the function is defined as the polynomial on
        [**r_min**, **r_max**] and zero outside it;
        0 ≤ **r_min** < **r_max** ≲ **max r**
        (**r_max** might exceed maximal **r**, but usually by < 1 pixel;
        negative **r_min** or **r_max** are allowed for convenience but are
        interpreted as 0)
    c: 2D numpy array
        polynomial coefficients for *r* and cos *θ* powers: ``c[m, n]`` is the
        coefficient for the :math:`r^m \cos^n\theta` term. This array can be
        conveniently constructed using :class:`Angular` tools.
    r_0 : float, optional
        *r* domain shift: the polynomial is defined in powers of
        (*r* − **r_0**) instead of *r*
    s : float, optional
        *r* stretching factor (around **r_0**): the polynomial is defined in
        powers of (*r* − **r_0**)/**s** instead of *r*
    """
    def __init__(self, r, cos, r_min, r_max, c, r_0=0.0, s=1.0):
        if r.shape != cos.shape:
            raise ValueError('Shapes of r and cos arrays must be equal.')

        # trim negative r limits
        if r_max <= 0:
            # both func and abel must be zero everywhere
            self.func = np.zeros_like(r)
            self.abel = np.zeros_like(r)
            return
        if r_min < 0:
            r_min = 0

        c = np.array(c, dtype=float)  # convert / make copy
        if np.ndim(c) != 2:
            raise ValueError('Coefficients array c must be 2-dimensional.')
        # highest cos power with non-zero coefficient
        N = c.nonzero()[1].max(initial=-1)
        if N < 0:  # all coefficients are zero
            # so both func and abel are also zero everywhere
            self.func = np.zeros_like(r)
            self.abel = np.zeros_like(r)
            return
        # for each cos power: highest r power with non-zero coefficient
        M = [a.nonzero()[0].max(initial=-1) for a in c.T]

        if s != 1.0:
            # apply stretch
            S = np.cumprod([1.0] + [1.0 / s] * max(M))  # powers of 1/s
            c *= np.array([S]).T
        if r_0 != 0.0:
            # apply shift
            m = max(M)
            P = pascal(1 + m, 'upper', False)  # binomial coefficients
            rm = np.cumprod([1.0] + [-float(r_0)] * m)  # powers of -r_0
            T = toeplitz([1.0] + [0.0] * m, rm)  # upper-diag. (-r_0)^{i - j}
            c = (P * T).dot(c)

        rfull, cosfull = r, cos  # (r and cos will be limited below)

        # Generate the polynomial function
        self.func = np.zeros_like(rfull)
        # limit calculations to relevant domain (outside it func = 0)
        dom = (r_min <= rfull) & (rfull < r_max)
        r = rfull[dom]
        cos = cosfull[dom]

        # sum all non-zero terms using Horner's method
        for n in range(N, -1, -1):
            if n < N:
                self.func[dom] *= cos
            if M[n] < 0:
                continue
            p = np.full_like(r, c[M[n], n])
            for m in range(M[n] - 1, -1, -1):
                p *= r
                if c[m, n]:
                    p += c[m, n]
            self.func[dom] += p

        # Generate its Abel transform
        self.abel = np.zeros_like(rfull)
        # relevant domain (outside it abel = 0)
        # (excluding r = 0 to avoid singularities, see below)
        dom = (0 < rfull) & (rfull < r_max)
        r = rfull[dom]
        cos = cosfull[dom]
        # values at lower and upper integration limits
        rho = [np.maximum(r, r_min),
               r_max]  # = max(r, r_max) within domain
        z = [np.sqrt(rho[0]**2 - r**2),
             np.sqrt(rho[1]**2 - r**2)]
        f = [np.minimum(r / r_min, 1.0) if r_min else 1.0,
             r / r_max]  # = min(r/r_max, 1) within domain

        # antiderivatives (recursive and used several times, thus cached)
        @cache
        def F(k, lim):  # lim: 0 = lower limit, 1 = upper limit
            if k < 0:
                return (z[lim] * f[lim]**k - k * F(k + 2, lim)) / (1 - k)
            if k == 0:
                return z[lim]
            if k == 1:
                return r * np.log(z[lim] + rho[lim])
            if k == 2:
                return r * np.arccos(f[lim])
            if k == 3:  # (using explicit expression for higher efficiency)
                return z[lim] * f[lim]
            # k > 3:  (in principle, k > 2)
            k -= 2
            return (z[lim] * f[lim]**k + (k - 1) * F(k, lim)) / k

        # sum all non-zero terms using Horner's method
        for n in range(N, -1, -1):
            if n < N:
                self.abel[dom] *= cos
            if M[n] < 0:
                continue
            p = c[M[n], n] * 2 * (F(n - M[n], 1) - F(n - M[n], 0))
            for m in range(M[n] - 1, -1, -1):
                p *= r
                if c[m, n]:
                    p += c[m, n] * 2 * (F(n - m, 1) - F(n - m, 0))
            self.abel[dom] += p
        # value at r = 0 (excluded above), nonzero only for n = 0
        dom = np.where(rfull == 0)
        for m in range(M[0] + 1):
            k = m + 1
            self.abel[dom] += c[m, 0] * 2 * (r_max**k - r_min**k) / k

        # help garbage collector to release cache memory
        F = None


class PiecewiseSPolynomial(BasePolynomial):
    r"""
    Piecewise bivariate polynomial function (sum of :class:`SPolynomial`\ s) in
    spherical coordinates and its Abel transform.

    Parameters
    ----------
    r, cos : numpy array
        *r* and cos *θ* values at which the function is generated
    ranges : iterable of unpackable
        (list of tuples of) polynomial parameters for each piece::

           [(r_min_1st, r_max_1st, c_1st),
            (r_min_2nd, r_max_2nd, c_2nd),
            ...
            (r_min_nth, r_max_nth, c_nth)]

        according to :class:`SPolynomial` conventions.
        All ranges are independent (may overlap and have gaps, may define
        polynomials of any degrees) and may include optional
        :class:`SPolynomial` parameters (``r_0, s``).
    """
    def __init__(self, r, cos, ranges):
        for rng in ranges:
            p = SPolynomial(r, cos, *rng)
            try:
                self.func += p.func
                self.abel += p.abel
            except AttributeError:  # first range
                self.func = p.func
                self.abel = p.abel


def rcos(rows=None, cols=None, shape=None, origin=None):
    r"""
    Create arrays with polar coordinates :math:`r` and :math:`\cos\theta`:
    either from a pair of Cartesian arrays (**rows**, **cols**) with row and
    column values for each point *or* for a uniform grid with given dimensions
    and origin (**shape**, **origin**).

    Parameters
    ----------
    rows, cols : numpy array
        arrays with respectively row and column values for each point. Must
        have identical shapes (the output arrays will have the same shape), but
        might contain any values. For example, can be 2D arrays with integer
        pixel coordinates, or with floating-point numbers for sampling at
        subpixel points or on a distorted grid, or 1D arrays for sampling along
        some curve.
    shape : tuple of int
        (rows, cols) -- create output arrays of given shape, with values
        corresponding to a uniform pixel grid.
    origin : tuple of float, optional
        position of the origin (:math:`r = 0`) in the output array. By default,
        the center of the array is used (center of the middle pixel for
        odd-sized dimensions; even-sized dimensions will have a corresponding
        half-pixel shift).

    Returns
    -------
    r : numpy array
        radii :math:`r = \sqrt{\text{row}^2 + \text{col}^2}` for each point
    cos : numpy array
        cosines of the polar angle :math:`\cos\theta = -\text{row}/r` for each
        point (by convention, :math:`\cos\theta = 0` at :math:`r = 0`)
    """
    if rows is not None or cols is not None:  # at least one array given
        # sanity checks:
        if rows is None or cols is None or \
           shape is not None or origin is not None:  # incompatible options
            raise ValueError('Arguments must be either '
                             'two arrays rows and cols or '
                             'shape=<tuple> and optional origin=<tuple>.')
        if rows.shape != cols.shape:
            raise ValueError('Shapes of rows and cols arrays must be equal.')
    else:
        # create rows and cols arrays for given shape
        rows, cols = np.mgrid[0.0:shape[0], 0.0:shape[1]]
        # prepare origin
        if origin is None:  # default = midpoint
            row = (shape[0] - 1) / 2
            col = (shape[1] - 1) / 2
        else:
            row, col = origin
            # to absolute coordinates
            if row < 0:
                row += shape[0]
            if col < 0:
                col += shape[1]
        # shift "0" to origin
        rows -= row
        cols -= col

    # radius
    r = np.sqrt(rows**2 + cols**2)
    # cosine
    cos = -np.divide(rows, r, where=r != 0, out=np.zeros_like(r))

    return r, cos


class Angular(object):
    r"""
    Class helping to define angular dependences for :class:`SPolynomial` and
    :class:`PiecewiseSPolynomial`.

    Supports arithmetic operations (addition, subtraction, multiplication of
    objects; multiplication and division by numbers) and outer product with
    radial coefficients (any list-like object). For example::

        [3, 0, -1] * (Angular.cos(4) + Angular.sin(4) / 2)

    represents :math:`(3 - r^2)\big(\cos^4\theta + (\sin^4\theta) / 2\big)`,
    producing ::

        [[ 1.5  0.  -3.  0.   4.5]
         [ 0.   0.  -0.  0.   0. ]
         [-0.5  0.   1.  0.  -1.5]]

    which can be supplied as the coefficient matrix to :class:`SPolynomial`.
    Likewise, a list of ranges for :class:`PiecewiseSPolynomial` can be
    prepared as an outer product with a list of ``(r_min, r_max, coeffs)``
    tuples (with optional other :class:`SPolynomial` parameters), where 1D
    ``coeffs`` contain radial coefficients for a polynomial segment.

    Parameters
    ----------
    c : float or iterable of float
        list of coefficients: ``Angular([c₀, c₁, c₂, ...])`` means
        :math:`c_0 \cos^0\theta + c_1 \cos^1\theta + c_2 \cos^2\theta + \dots`;
        ``Angular(a)`` represents the isotropic distribution a⋅cos⁰ *θ*

    Attributes
    ----------
    c : numpy array
        coefficients for :math:`\cos^n\theta` powers, passed at instantiation
        directly (see above) or converted from other representations by the
        methods below.
    """
    def __init__(self, c):
        """
        Weighted sum of cosine powers.
        """
        self.c = np.ravel(c).astype(float)

    @classmethod
    def cos(cls, n):
        r"""
        Cosine power: ``Angular.cos(n)`` means :math:`\cos^n\theta`.
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError('Power must be positive integer.')
        return cls([0] * n + [1])

    @classmethod
    def sin(cls, n):
        r"""
        Sine power: ``Angular.sin(n)`` means :math:`\sin^n\theta`
        (*n* must be even).
        """
        return cls.cossin(0, n)

    @classmethod
    def cossin(cls, m, n):
        r"""
        Product of cosine and sine powers: ``Angular.cossin(m, n)`` means
        :math:`\cos^m\theta \cdot \sin^n\theta` (*n* must be even).
        """
        if not isinstance(m, int) or m < 0:
            raise ValueError('Cosine power must be positive integer.')
        if not isinstance(n, int) or n < 0 or n % 2:
            raise ValueError('Sine power must be even positive integer.')
        c = np.zeros(1 + m + n)
        # binomial coefficients of (1 - cos^2)^(n/2)
        c[m::2] = invpascal(1 + n // 2, 'lower', False)[-1, ::-1]
        return cls(c)

    @classmethod
    def legendre(cls, c):
        r"""
        Weighted sum of Legendre polynomials in cos *θ*:
        ``Angular.legendre([c₀, c₁, c₂, ...])`` means
        :math:`c_0 P_0(\cos\theta) + c_1 P_1(\cos\theta) + c_2 P_2(\cos\theta)
        + \dots`

        This method is intended to be called like ::

            Angular.legendre([1, β₁, β₂, ...])

        where :math:`\beta_i` are so-called anisotropy parameters. However, if
        you really need a single polynomial :math:`P_n(\cos\theta)`, this can
        be easily achieved by ::

            Angular.legendre([0] * n + [1])

        """
        C = np.zeros_like(c, dtype=float)
        for n, a in enumerate(c):
            C[n::-2] += a * legendre(n).c[::2]
            # (SciPy's legendre() has backwards order and produces noise in
            #  coefficients that must be zero, so indexing takes care of this)
        return cls(C)

    # disable NumPy "ufunc" handling, which makes no sense here
    # and interferes with the overloaded multiplication operator
    __array_ufunc__ = None

    def __add__(self, other):
        """
        Sum of two objects (might have different sizes).
        """
        a, b, = sorted([self.c, other.c], key=len)
        c = b.copy()  # copy the longer array
        c[:len(a)] += a  # add the shorter array to the relevant part
        return Angular(c)

    def __sub__(self, other):
        """
        Difference of two objects (might have different orders).
        """
        a, b, = sorted([self.c, other.c], key=len)
        c = b.copy()  # copy the longer array
        c[:len(a)] -= a  # subtract the shorter array from the relevant part
        return Angular(c)

    def __mul__(self, obj):
        """
        Multiplication by number or another angular dependence: return the
        resulting angular dependence.

        Outer product of radial and angular coefficients: return 2D array with
        rows corresponding to powers of *r* and columns to powers of cos *θ*.
        """
        if isinstance(obj, Angular):  # by another angular dependence
            return Angular(np.convolve(self.c, obj.c))
        if np.isscalar(obj):  # by number
            return Angular(obj * self.c)
        try:  # piecewise ranges
            ranges = []
            for rng in obj:
                r_min, r_max, c = rng[:3]
                c = np.outer(np.ravel(c), self.c)
                rng = (r_min, r_max, c) + rng[3:]
                ranges.append(rng)
            return ranges
        except TypeError:  # otherwise -- by radial coefficients
            return np.outer(np.ravel(obj), self.c)

    __rmul__ = __mul__
    __rmul__.__doc__ = __mul__.__doc__

    def __truediv__(self, num):
        """
        Division by number.
        """
        return Angular(self.c / num)

    def __repr__(self):
        return str(self.c) + '.[cos^n]'


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


def bspline(spl):
    """
    Convert SciPy B-spline to :class:`PiecewisePolynomial` parameters.

    Parameters
    ----------
    spl : tuple or BSpline or UnivariateSpline
        ``scipy.interpolate`` B-spline representation, such as ``splrep()``
        results, ``BSpline`` object (result of ``make_interp_spline()``, for
        example) or ``UnivariateSpline`` object

    Returns
    -------
    ranges : list of tuple
        list of parameters ``(r_min, r_max, coeffs, r_0)`` that can be passed
        directly to :class:`PiecewisePolynomial` or, after “multiplication” by
        :class:`Angular`, to :class:`PiecewiseSPolynomial`
    """
    if isinstance(spl, UnivariateSpline):
        # extract necessary data, convert to compatible format
        knots = spl.get_knots()
        coeffs = spl.get_coeffs()
        k = len(coeffs) - len(knots) + 1
        knots = np.pad(knots, k, 'edge')
        spl = (knots, coeffs, k)

    # convert B-spline representation to piecewise polynomial representation
    ppoly = PPoly.from_spline(spl)
    x = ppoly.x  # breakpoints
    c = ppoly.c.T[:, ::-1]  # coefficients (PPoly degrees are descending)

    # convert to PiecewisePolynomial ranges
    ranges = []
    for i in range(len(x) - 1):
        r_min, r_max = x[i], x[i + 1]
        if r_min != r_max:  # (some PPoly intervals are degenerate)
            ranges.append((r_min, r_max, c[i], r_min))
    return ranges
