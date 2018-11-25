# -*- coding: utf-8 -*-

from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import hermite
from math import factorial

import abel
from abel.tools.polynomial import Polynomial, PiecewisePolynomial


def test_polynomial_shape():
    """
    Testing that func and abel have the same shape as r.
    """
    n = 20
    r = np.linspace(0, n/2, n)
    P = Polynomial(r, 0, n, [1])

    assert P.func.shape == r.shape
    assert P.abel.shape == r.shape


def test_polynomial_zeros():
    """
    Testing that zero coefficients produce zero func and abel.
    """
    n = 20
    r = np.linspace(0, n/2, n)
    P = Polynomial(r, 0, n, [0, 0])

    assert_allclose(P.func, 0)
    assert_allclose(P.abel, 0)


def test_polynomial_step():
    """
    Testing step function (and multiplication).
    """
    n = 50
    r = np.linspace(0, n/2, n)
    r_min = 10
    r_max = 20
    A = 2

    P = A * Polynomial(r, r_min, r_max, [1/A])

    func = np.zeros_like(r)
    func[(r >= r_min) * (r <= r_max)] = 1

    def sqrt0(x): return np.sqrt(x, np.zeros_like(x), where=x > 0)
    abel = 2 * (sqrt0(r_max**2 - r**2) - sqrt0(r_min**2 - r**2))

    assert_allclose(P.func, func)
    assert_allclose(P.abel, abel)


def test_polynomial_gaussian():
    """
    Testing shifted and scaled Taylor expansion of a Gaussian,
    in reduced coordinates.
    """
    n = 100
    r = np.linspace(0, n/2, n)
    sigma = 15

    # coefficients of Taylor series around r = 1
    c = []
    for k in range(50):
        c.append(np.exp(-1.0) * hermite(k)(-1.0) / factorial(k))

    P = Polynomial(r, 0, r[-1] + 2, c, sigma, sigma, reduced=True)
    # scaling and shifting by sigma should produce exp(-(r/σ)^2),
    # its Abel transform is √π σ exp(-(r/σ)^2)

    func = np.exp(-(r / sigma)**2)
    abel = np.sqrt(np.pi) * sigma * func

    assert_allclose(P.func, func, atol=1.0e-7)
    assert_allclose(P.abel, abel, atol=1.0e-4)


def test_polynomial_smoothstep():
    """
    Testing cubic smoothstep function,
    using ``direct`` inverse transform.
    """
    n = 100
    r = np.arange(float(n))
    r_min = 20
    r_max = 80
    w = 10
    h = 3

    c = [1/2, 3/4, 0, -1/4]
    P = PiecewisePolynomial(r, [(r_min - w, r_min + w, c, r_min, w),
                                (r_min + w, r_max - w, [1]),
                                (r_max - w, r_max + w, c, r_max, -w)])
    P *= h

    recon = abel.direct.direct_transform(P.abel, backend='python')

    assert_allclose(P.func, recon, atol=2.0e-2)


if __name__ == '__main__':
    test_polynomial_shape()
    test_polynomial_zeros()
    test_polynomial_step()
    test_polynomial_gaussian()
    test_polynomial_smoothstep()
