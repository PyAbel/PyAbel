# -*- coding: utf-8 -*-
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import hermite
from scipy.interpolate import splrep, splev, make_interp_spline, UnivariateSpline
from math import factorial

import abel
from abel.tools.polynomial import Polynomial, PiecewisePolynomial, \
    rcos, Angular, SPolynomial, PiecewiseSPolynomial, ApproxGaussian, bspline


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


def test_polynomial_copy():
    """
    Test copy creation.
    """
    n = 20
    r = np.arange(n, dtype=float)
    P0 = Polynomial(r, 0, n/2, [1, -1], 0, n/2)
    P0c = P0.copy()

    # should not change P0 and P0c
    P1 = 2 * P0
    assert_allclose(P0.func, P0c.func)
    assert_allclose(P0.abel, P0c.abel)
    assert_allclose(P1.func, (P0 * 2).func)
    assert_allclose(P1.abel, (P0 * 2).abel)

    # should change P0, but not P0c
    P0 *= 2
    assert_allclose(P0.func, P1.func)
    assert_allclose(P0.abel, P1.abel)
    assert_allclose(P0.func, (P0c * 2).func)
    assert_allclose(P0.abel, (P0c * 2).abel)


def test_ppolynomial_copy():
    """
    Test copy creation.
    """
    n = 20
    r = np.arange(n, dtype=float)
    P0 = PiecewisePolynomial(r, [(0, n/2, [0, 1], 0, n/2),
                                 (n/2, n, [1, -1], n/2, n/2)])
    assert(len(P0.p) == 2)
    P0c = P0.copy()
    assert(len(P0c.p) == len(P0.p))

    # should not change P0 and P0c
    P1 = 2 * P0
    assert(len(P1.p) == len(P0.p))
    assert_allclose(P0.func, P0c.func)
    assert_allclose(P0.abel, P0c.abel)
    assert_allclose(P1.func, (P0 * 2).func)
    assert_allclose(P1.abel, (P0 * 2).abel)
    for i in range(len(P0.p)):
        assert_allclose(P0.p[i].func, P0c.p[i].func)
        assert_allclose(P0.p[i].abel, P0c.p[i].abel)
        assert_allclose(P1.p[i].func, (P0.p[i] * 2).func)
        assert_allclose(P1.p[i].abel, (P0.p[i] * 2).abel)

    # should change P0, but not P0c
    P0 *= 2
    assert(len(P0.p) == len(P0c.p))
    assert_allclose(P0.func, P1.func)
    assert_allclose(P0.abel, P1.abel)
    assert_allclose(P0.func, (P0c * 2).func)
    assert_allclose(P0.abel, (P0c * 2).abel)
    for i in range(len(P0.p)):
        assert_allclose(P0.p[i].func, P1.p[i].func)
        assert_allclose(P0.p[i].abel, P1.p[i].abel)
        assert_allclose(P0.p[i].func, (P0c.p[i] * 2).func)
        assert_allclose(P0.p[i].abel, (P0c.p[i] * 2).abel)


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
    c = [np.exp(-1.0) * hermite(k)(-1.0) / factorial(k) for k in range(50)]

    P = Polynomial(r, 0, r[-1] + 2, c, sigma, sigma, reduced=True)
    # scaling and shifting by sigma should produce exp(-(r/σ)²),
    # its Abel transform is √π σ exp(-(r/σ)²)

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


def test_approx_gaussian():
    """
    Test Gaussian approximation against exact Gaussian.
    """
    # reference Gaussian
    def g(x):
        return np.exp(-x**2 / 2)

    # test tolerances
    r = np.linspace(0, 5, 1000)
    ref = g(r)
    for tol in [5e-2, 1e-2, 1e-3, 1e-4, 1e-6]:
        ag = ApproxGaussian(tol)
        P = PiecewisePolynomial(r, ag.ranges)
        assert_allclose(P.func, ref, atol=tol, err_msg='-> tol={}'.format(tol))

    # test scaling
    r = np.arange(100, dtype=float)
    A = 2
    r0 = 30
    sigma = 20
    ref = A * g((r - r0) / sigma)
    ag = ApproxGaussian()
    P = PiecewisePolynomial(r, ag.scaled(A, r0, sigma))
    assert_allclose(P.func, ref, atol=A * 5e-3)


def test_bspline():
    """
    Test B-spline conversion.
    """
    # data for fitting
    x = np.arange(101)
    y = np.exp(-((x - 50)/15)**2)
    # points for output sampling
    r = np.linspace(0, 200, 1000)

    # from tck tuple
    for k in [1, 2, 3, 4, 5]:  # all supported degrees
        spl = splrep(x, y, k=k, s=0.1)
        ref = splev(r, spl, ext=1)  # (0 outside)
        P = PiecewisePolynomial(r, bspline(spl))
        assert_allclose(P.func, ref, err_msg='-> splrep, k={}'.format(k))

    # from BSpline
    for k in [0, 1, 2, 3, 5, 7]:  # all supported degrees up to 7
        # clamped boundary conditions
        if k < 2:
            bc = None
        elif k == 2:
            bc = (None, [(1, 0)])
        else:
            bc = ([(i, 0) for i in range(1, (k + 1) // 2)],) * 2
        # spline through decimated data
        spl = make_interp_spline(x[::10], y[::10], k=k, bc_type=bc)
        ref = spl(r, extrapolate=False)
        ref[np.isnan(ref)] = 0  # (0 outside)
        P = PiecewisePolynomial(r, bspline(spl))
        assert_allclose(P.func, ref, atol=1e-10,
                        err_msg='-> make_interp_spline, k={}'.format(k))

    # from UnivariateSpline
    for k in [1, 2, 3, 4, 5]:  # all supported degrees up to 7
        spl = UnivariateSpline(x, y, k=k, s=0.1, ext='zeros')
        ref = spl(r)
        P = PiecewisePolynomial(r, bspline(spl))
        assert_allclose(P.func, ref,
                        err_msg='-> UnivariateSpline, k={}'.format(k))


def test_rcos():
    """
    Testing rcos shape and origin.
    """
    # 1D
    rows = np.array([0, 0, -1, 1])
    cols = np.array([0, 1,  1, 0])
    r, cos = rcos(rows, cols)
    assert(r.shape == cos.shape == rows.shape)
    assert_allclose(r, [0, 1, np.sqrt(2), 1])
    assert_allclose(cos, [0, 0, np.sqrt(1/2), -1])

    n = 5
    r, cos = rcos(shape=(n, n))
    # shape
    assert(r.shape == cos.shape == (5, 5))
    # default origin
    assert(r[n // 2, n // 2] == 0)
    assert(r[0, 0] == r[0, -1] == r[-1, -1] == r[-1, 0] != 0)
    assert(cos[0, n // 2] == 1)
    assert(cos[-1, n // 2] == -1)
    assert(cos[n // 2, 0] == cos[n // 2, -1] == 0)
    assert(cos[0, 0] == cos[0, -1] == -cos[-1, -1] == -cos[-1, 0] != 0)

    r, cos = rcos(shape=(n, n), origin=(0, 0))
    # origin at upper left corner
    assert(r[0, 0] == 0)
    assert(r[0, -1] == r[-1, 0] != 0)
    assert(cos[-1, 0] == -1)
    assert(cos[0, -1] == 0)

    r, cos = rcos(shape=(n, n), origin=(-1, 0))
    # origin at lower left corner
    assert(r[-1, 0] == 0)
    assert(r[0, 0] == r[-1, -1] != 0)
    assert(cos[0, 0] == 1)
    assert(cos[-1, 0] == 0)

    r, cos = rcos(shape=(n, n), origin=(0, -1))
    # origin at upper right corner
    assert(r[0, -1] == 0)
    assert(r[0, 0] == r[-1, -1] != 0)
    assert(cos[-1, -1] == -1)
    assert(cos[0, 0] == 0)


def test_angular():
    """
    Test Angular class.
    """
    # init isotropic
    A = Angular(2)
    assert A.c == [2.0]

    # multiplication of objects
    Ac = Angular.cos(3)
    As = Angular.sin(4)
    Acs = Angular.cossin(3, 4)
    assert_allclose((Ac * As).c, Acs.c)

    # Legendre, multiplication by number, subtraction
    Al = Angular.legendre([0] * 2 + [1])  # P₂
    Aref = 3/2 * Angular.cos(2) - Angular(1/2)
    assert_allclose(Al.c, Aref.c, err_msg='-> P_2')

    Al = Angular.legendre([0] * 3 + [1])  # P₃
    Aref = 5/2 * Angular.cos(3) - 3/2 * Angular.cos(1)
    assert_allclose(Al.c, Aref.c, err_msg='-> P_3')

    # Legendre as anisotropy parameters
    Al = Angular.legendre([1, 0, -1]) * 2/3  # β = −1 ⇒ sin²
    Aref = Angular.sin(2)
    assert_allclose(Al.c, Aref.c, err_msg='-> sin^2')

    Al = Angular.legendre([1, 0, +2]) * 1/3  # β = +2 ⇒ cos²
    Aref = Angular.cos(2)
    assert_allclose(Al.c, Aref.c, err_msg='-> cos^2')

    # division, addition, outer product
    c = [3, 0, -1] * (Angular.cos(4) + Angular.sin(4) / 2)
    ref = [[ 1.5, 0, -3, 0,  4.5],
           [ 0.0, 0, -0, 0,  0.0],
           [-0.5, 0,  1, 0, -1.5]]
    assert_allclose(c, ref)


def test_spolynomial_shape():
    """
    Testing that func and abel have the same shape as r.
    """
    # 2D
    n = 20
    r, cos = rcos(shape=(2 * n + 1, 2 * n + 1))
    P = SPolynomial(r, cos, 0, n, [[1]])
    assert P.func.shape == r.shape
    assert P.abel.shape == r.shape

    # 1D
    r = np.arange(10, dtype=float)
    cos = np.ones_like(r)
    P = SPolynomial(r, cos, 0, n, [[1]])
    assert P.func.shape == r.shape
    assert P.abel.shape == r.shape


def test_spolynomial_zeros():
    """
    Testing that zero coefficients produce zero func and abel.
    """
    n = 20
    r, cos = rcos(shape=(2 * n + 1, 2 * n + 1))
    P = SPolynomial(r, cos, 0, n, [[0, 0], [0, 0]])
    assert_allclose(P.func, 0)
    assert_allclose(P.abel, 0)


def test_spolynomial_copy():
    """
    Test copy creation.
    """
    n = 20
    r, cos = rcos(shape=(2 * n + 1, 2 * n + 1))
    P0 = SPolynomial(r, cos, 0, n, [[1, 0, -1],
                                    [-1, 0, 1]], 0, n)
    P0c = P0.copy()

    # should not change P0 and P0c
    P1 = 2 * P0
    assert_allclose(P0.func, P0c.func)
    assert_allclose(P0.abel, P0c.abel)
    assert_allclose(P1.func, (P0 * 2).func)
    assert_allclose(P1.abel, (P0 * 2).abel)

    # should change P0, but not P0c
    P0 *= 2
    assert_allclose(P0.func, P1.func)
    assert_allclose(P0.abel, P1.abel)
    assert_allclose(P0.func, (P0c * 2).func)
    assert_allclose(P0.abel, (P0c * 2).abel)


def test_spolynomial_gaussian():
    """
    Testing Gaussian approximation with spherical polynomials.
    """
    # reference Gaussian
    def g(r):
        return np.exp(-r**2 / 2)
    n = 201
    sigma = 20
    r, cos = rcos(shape=(n, n))
    ref_func = g(r / sigma)
    mul = np.sqrt(2 * np.pi) * sigma
    ref_abel = mul * ref_func
    for tol in [5e-2, 1e-2, 1e-3, 1e-4, 1e-6]:
        coef = Angular(1) * ApproxGaussian(tol).scaled(1, 0, sigma)
        P = PiecewiseSPolynomial(r, cos, coef)
        assert_allclose(P.func, ref_func, atol=1.001 * tol,  # with small slack
                        err_msg='-> func, tol={}'.format(tol))
        assert_allclose(P.abel, ref_abel, atol=0.85 * mul * tol,  # somewhat
                        err_msg='-> abel, tol={}'.format(tol))    # better


def test_spolynomial_high():
    """
    Testing higher orders of spherical polynomials
    using ``direct`` forward transform.
    """
    n = 201
    r, cos = rcos(shape=(n, n))

    # smooth peak (1 − r)³ (1 + r)³ for −1 ≤ r ≤ +1
    peak = [1, 0, -3, 0, 3, 0, -1]
    r0 = 70
    w = 20

    for n in range(9):
        P = SPolynomial(r, cos, r0 - w, r0 + w,
                        peak * Angular([0] * n + [1]), r0, w)
        ref = abel.Transform(P.func, direction='forward', method='direct',
                             symmetry_axis=0,
                             transform_options={'backend': 'Python'}).transform
        assert_allclose(P.abel, ref,
                        atol=np.max(ref) * 5.4e-3,  # 'direct' adds some error
                        err_msg='-> n={}'.format(n))


if __name__ == '__main__':
    test_polynomial_shape()
    test_polynomial_zeros()
    test_polynomial_copy()
    test_ppolynomial_copy()
    test_polynomial_step()
    test_polynomial_gaussian()
    test_polynomial_smoothstep()

    test_approx_gaussian()
    test_bspline()

    test_rcos()
    test_angular()
    test_spolynomial_shape()
    test_spolynomial_zeros()
    test_spolynomial_copy()
    test_spolynomial_gaussian()
    test_spolynomial_high()
