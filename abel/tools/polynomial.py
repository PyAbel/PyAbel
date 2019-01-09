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


class Polynomial(object):
    """
    Polynomial function and its Abel transform.

    Supports multiplication and division by numbers.

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
        (**r_max** might exceed maximal **r**, but usually by < 1 pixel)
    c: numpy array
        polynomial coefficients in order of increasing degree:
        [c₀, c₁, c₂] means c₀ + c₁ *r* + c₂ *r*\ ²
    r_0 : float, optional
        origin shift: the polynomial is defined as
        c₀ + c₁ (*r* − **r_0**) + c₂ (*r* − **r_0**)² + ...
    s : float, optional
        *r* stretching factor (around **r_0**): the polynomial is defined as
        c₀ + c₁ (*r*/**s**) + c₂ (*r*/**s**)² + ...
    reduced : boolean, optional
        internally rescale the *r* range to [0, 1];
        useful to avoid floating-point overflows for high degrees
        at large r (and might improve numeric accuracy)
    """
    def __init__(self, r, r_min, r_max, c, r_0=0.0, s=1.0, reduced=False):
        # remove zero high-order terms
        c = np.array(np.trim_zeros(c, 'b'), float)
        # if all coefficients are zero
        if len(c) == 0:
            # then both func and abel are also zero everywhere
            self.func = np.zeros_like(r)
            self.abel = self.func
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
        n = r.shape[0]
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

    def __imul__(self, m):
        """
        In-place multiplication: Polynomial *= num.
        """
        self.func *= m
        self.abel *= m
        return self

    def __mul__(self, m):
        """
        Multiplication: Polynomial * num.
        """
        res = self.__new__(type(self))  # create empty object (same type)
        res.func = self.func * m
        res.abel = self.abel * m
        return res

    __rmul__ = __mul__
    __rmul__.__doc__ = \
        """
        Multiplication: num * Polynomial.
        """

    def __itruediv__(self, m):
        """
        In-place division: Polynomial /= num.
        """
        return self.__imul__(1 / m)

    def __truediv__(self, m):
        """
        Division: Polynomial / num.
        """
        return self.__mul__(1 / m)

    # (Addition and subtraction are not implemented because they are
    #  meaningful only for polynomials generated over the same r array.
    #  Use PiecewisePolynomial for sums of polynomials.)


class PiecewisePolynomial(Polynomial):
    """
    Piecewise polynomial function (sum of ``Polynomial``\ s)
    and its Abel transform.

    Supports multiplication and division by numbers.

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
    """
    def __init__(self, r, ranges):
        self.p = []
        for rng in ranges:
            self.p.append(Polynomial(r, *rng))

        func = self.p[0].func
        for p in self.p[1:]:
            func += p.func
        self.func = func

        abel = self.p[0].abel
        for p in self.p[1:]:
            abel += p.abel
        self.abel = abel

    # (Multiplication and division methods are inherited from Polynomial.)
