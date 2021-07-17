from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import inv, toeplitz

import abel
from abel.daun import _bs_daun, get_bs_cached, cache_cleanup, daun_transform
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


def test_daun_basis_sets_cache():
    """Check basis-set cache handling"""

    # changing size
    n1, n2 = 20, 30
    for order in range(4):
        # forward
        cache_cleanup()
        bs2_ref = get_bs_cached(n2, order, direction='forward')
        cache_cleanup()
        bs1_ref = get_bs_cached(n1, order, direction='forward')
        bs2 = get_bs_cached(n2, order, direction='forward')
        bs1 = get_bs_cached(n1, order, direction='forward')
        assert_allclose(bs1, bs1_ref, atol=1e-15,
                        err_msg='-> forward: n<, order = {}'.format(order))
        assert_allclose(bs2, bs2_ref, atol=1e-15,
                        err_msg='-> forward: n>, order = {}'.format(order))

        # inverse
        for reg_type in [None, 'diff', 'L2', 'nonneg']:
            cache_cleanup()
            bs2_ref = get_bs_cached(n2, order, reg_type, 1)
            cache_cleanup()
            bs1_ref = get_bs_cached(n1, order, reg_type, 1)
            bs2 = get_bs_cached(n2, order, reg_type, 1)
            bs1 = get_bs_cached(n1, order, reg_type, 1)
            assert_allclose(bs1, bs1_ref, atol=1e-15,
                            err_msg='-> inverse: n<, order = {}, reg_type = {}'.
                                    format(order, reg_type))
            assert_allclose(bs2, bs2_ref, atol=1e-15,
                            err_msg='-> inverse: n>, order = {}, reg_type = {}'.
                                    format(order, reg_type))

    # changing order
    for direction in ['forward', 'inverse']:
        cache_cleanup()
        bs0_ref = get_bs_cached(n1, 0)
        cache_cleanup()
        bs1_ref = get_bs_cached(n1, 1)
        bs0 = get_bs_cached(n1, 0)
        bs1 = get_bs_cached(n1, 1)
        assert_allclose(bs0, bs0_ref, atol=1e-15,
                        err_msg='-> {}: order>'.
                                format(direction, order, reg_type))
        assert_allclose(bs1, bs1_ref, atol=1e-15,
                        err_msg='-> {}: order<'.
                                format(direction, order, reg_type))

    # changing regularization strength
    bs_ref = []
    for reg in [0, 1, 2]:
        cache_cleanup()
        bs_ref.append(get_bs_cached(n1, strength=reg))
    bs = []
    for reg in [0, 1, 2]:
        bs.append(get_bs_cached(n1, strength=reg))
    assert_allclose(bs[0], bs_ref[0], atol=1e-15, err_msg='-> reg = 0 after 2')
    assert_allclose(bs[1], bs_ref[1], atol=1e-15, err_msg='-> reg = 1 after 0')
    assert_allclose(bs[2], bs_ref[2], atol=1e-15, err_msg='-> reg = 2 after 1')


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
                     [('L2', 1), 6e-4],  # for r > 0
                     [('L2c', 1), 6e-4],
                     ['nonneg', 2e-3]):
        recon = daun_transform(tr, reg=reg, verbose=False)
        recon = recon[n // 2 + n % 2]
        skip = 1 if isinstance(reg, tuple) and reg[0] == 'L2' else 0
        assert_allclose(recon[skip:], ref.func[skip:], atol=tol,
                        err_msg='-> reg = ' + repr(reg))

    # test dr
    recon = daun_transform(tr, dr=0.5, verbose=False)[0]
    assert_allclose(recon, 2 * ref.func, atol=3e-3, err_msg='-> dr = 0.5')


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

    # test dr
    proj = daun_transform(tr, dr=0.5, direction='forward', verbose=False)[0]
    assert_allclose(proj, ref.abel / 2, atol=2e-2, err_msg='-> dr = 0.5')


if __name__ == '__main__':
    test_daun_basis_sets()
    test_daun_basis_sets_cache()
    test_daun_shape()
    test_daun_zeros()
    test_daun_gaussian()
    test_daun_forward_gaussian()
