from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import inv, toeplitz

import abel
from abel.daun import _bs_daun, daun_transform
from abel.tools.polynomial import PiecewisePolynomial as PP
from abel.tools.analytical import GaussianAnalytical


def daun_basis_set(n, order):
    """Reference implementation of basis-set generation"""
    r = np.arange(float(n))

    A = np.empty((n, n))
    for j in range(n):
        if order == 0:
            p = PP(r, [(j - 1/2, j + 1/2, [1], j)])
        elif order == 1:
            p = PP(r, [(j - 1, j, [1,  1], j),
                       (j, j + 1, [1, -1], j)])
        elif order == 2:
            p = PP(r, [(j - 1,   j - 1/2, [0, 0,  2], j - 1),
                       (j - 1/2, j + 1/2, [1, 0, -2], j),
                       (j + 1/2, j + 1,   [0, 0,  2], j + 1)])
        else:  # order == 3:
            p = PP(r, [(j - 1, j, [1, 0, -3, -2], j),
                       (j, j + 1, [1, 0, -3,  2], j)])
        A[j] = p.abel

    if order == 3:
        B = np.empty((n, n))
        for j in range(n):
            B[j] = PP(r, [(j - 1, j, [0, 1,  2, 1], j),
                          (j, j + 1, [0, 1, -2, 1], j)]).abel
        C = toeplitz([4, 1] + [0] * (n - 2))
        C[1, 0] = C[-2, -1] = 0
        D = toeplitz([0, 3] + [0] * (n - 2), [0, -3] + [0] * (n - 2))
        D[1, 0] = D[-2, -1] = 0
        A += D.dot(inv(C)).dot(B)

    return A


def test_daun_basis_sets():
    """Check basis-set generation for all orders"""
    n = 10
    for order in range(4):
        A = _bs_daun(n, order)
        Aref = daun_basis_set(n, order)
        assert_allclose(A, Aref, err_msg='-> order = ' + str(order))


def test_daun_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')
    recon = daun_transform(x, verbose=False)
    assert recon.shape == (n, n)


def test_daun_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')
    recon = abel.daun.daun_transform(x, verbose=False)
    assert_allclose(recon, 0)


def test_daun_gaussian():
    """Check inverse Daun transform of a gaussian"""
    n = 100
    r_max = n - 1

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=30)
    tr = np.tile(ref.abel[None, :], (n, 1))  # make a 2D array from 1D

    for order, tol in [(0, 2e-3), (1, 3e-4), (2, 9e-4), (3, 4e-5)]:
        recon = daun_transform(tr, order=order, verbose=False)
        recon = recon[n // 2 + n % 2]
        assert_allclose(recon, ref.func, atol=tol,
                        err_msg='-> order = ' + str(order))

    for reg, tol in ([None, 2e-3],
                     [1, 1e-3],
                     [('diff', 1), 1e-3],
                     [('L2', 1), 6e-4],
                     ['nonneg', 2e-3]):
        recon = daun_transform(tr, reg=reg, verbose=False)
        recon = recon[n // 2 + n % 2]
        skip = 1 if isinstance(reg, tuple) and reg[0] == 'L2' else 0
        assert_allclose(recon[skip:], ref.func[skip:], atol=tol,
                        err_msg='-> reg = ' + repr(reg))


def test_daun_forward_gaussian():
    """Check forward Daun transform of a gaussian"""
    n = 100
    r_max = n - 1

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=30)
    tr = np.tile(ref.func[None, :], (n, 1))  # make a 2D array from 1D

    for order, tol in [(0, 3e-2), (1, 7e-3), (2, 2e-2), (3, 7e-4)]:
        proj = daun_transform(tr, order=order, direction='forward',
                              verbose=False)
        proj = proj[n // 2 + n % 2]
        assert_allclose(proj, ref.abel, atol=tol,
                        err_msg='-> order = ' + str(order))


if __name__ == '__main__':
    test_daun_basis_sets()
    test_daun_shape()
    test_daun_zeros()
    test_daun_gaussian()
    test_daun_forward_gaussian()
