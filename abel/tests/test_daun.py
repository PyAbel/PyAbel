from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import abel
from abel.daun import daun_transform
from abel.tools.analytical import GaussianAnalytical


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
    test_daun_shape()
    test_daun_zeros()
    test_daun_gaussian()
    test_daun_forward_gaussian()
