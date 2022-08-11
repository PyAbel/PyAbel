from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose
from scipy.linalg import inv, toeplitz

import abel
from abel.daun import _bs_daun, get_bs_cached, cache_cleanup, daun_transform
from abel.tools.polynomial import PiecewisePolynomial as PP
from abel.tools.analytical import GaussianAnalytical


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')


def daun_bs(n, degree):
    """Reference implementation of basis-set generation"""
    r = np.arange(float(n))

    A = np.empty((n, n))
    for j in range(n):
        if degree == 0:
            p = PP(r, [(j - 1/2, j + 1/2, [1], j)])
        elif degree == 1:
            p = PP(r, [(j - 1, j, [1,  1], j),
                       (j, j + 1, [1, -1], j)])
        elif degree == 2:
            p = PP(r, [(j - 1,   j - 1/2, [0, 0,  2], j - 1),
                       (j - 1/2, j + 1/2, [1, 0, -2], j),
                       (j + 1/2, j + 1,   [0, 0,  2], j + 1)])
        else:  # degree == 3:
            p = PP(r, [(j - 1, j, [1, 0, -3, -2], j),
                       (j, j + 1, [1, 0, -3,  2], j)])
        A[j] = p.abel

    if degree == 3:
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


def test_daun_bs():
    """Check basis-set generation for all degrees"""
    n = 10
    for degree in range(4):
        A = _bs_daun(n, degree)
        Aref = daun_bs(n, degree)
        assert_allclose(A, Aref, err_msg='-> degree = ' + str(degree))


def test_daun_bs_cache():
    """Check basis-set cache handling"""

    # changing size
    n1, n2 = 20, 30
    for degree in range(4):
        # forward
        cache_cleanup()
        bs2_ref = get_bs_cached(n2, degree, direction='forward')
        cache_cleanup()
        bs1_ref = get_bs_cached(n1, degree, direction='forward')
        bs2 = get_bs_cached(n2, degree, direction='forward')
        bs1 = get_bs_cached(n1, degree, direction='forward')
        assert_allclose(bs1, bs1_ref, atol=1e-15,
                        err_msg='-> forward: n<, degree = {}'.format(degree))
        assert_allclose(bs2, bs2_ref, atol=1e-15,
                        err_msg='-> forward: n>, degree = {}'.format(degree))

        # inverse
        for reg_type in [None, 'diff', 'L2', 'nonneg']:
            cache_cleanup()
            bs2_ref = get_bs_cached(n2, degree, reg_type, 1)
            cache_cleanup()
            bs1_ref = get_bs_cached(n1, degree, reg_type, 1)
            bs2 = get_bs_cached(n2, degree, reg_type, 1)
            bs1 = get_bs_cached(n1, degree, reg_type, 1)
            assert_allclose(bs1, bs1_ref, atol=1e-15,
                            err_msg='-> inverse: n<, degree = {}, reg_type = {}'.
                                    format(degree, reg_type))
            assert_allclose(bs2, bs2_ref, atol=1e-15,
                            err_msg='-> inverse: n>, degree = {}, reg_type = {}'.
                                    format(degree, reg_type))

    # changing degree
    for direction in ['forward', 'inverse']:
        cache_cleanup()
        bs0_ref = get_bs_cached(n1, 0)
        cache_cleanup()
        bs1_ref = get_bs_cached(n1, 1)
        bs0 = get_bs_cached(n1, 0)
        bs1 = get_bs_cached(n1, 1)
        assert_allclose(bs0, bs0_ref, atol=1e-15,
                        err_msg='-> {}: degree>'.format(direction))
        assert_allclose(bs1, bs1_ref, atol=1e-15,
                        err_msg='-> {}: degree<'.format(direction))

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


def get_basis_file_name(n, degree):
    return os.path.join(DATA_DIR, 'daun_basis_{}_{}.npy'.format(n, degree))


def test_daun_bs_disk_cache():
    """Check basis-set disk caching"""
    n1, n2 = 10, 20

    f_n1_0 = get_basis_file_name(n1, 0)
    f_n2_0 = get_basis_file_name(n2, 0)
    f_n1_1 = get_basis_file_name(n1, 1)

    for fname in [f_n1_0, f_n2_0, f_n1_1]:
        if os.path.exists(fname):
            os.remove(fname)

    # saving
    cache_cleanup()
    bs_n1_0 = get_bs_cached(n1, 0, basis_dir=DATA_DIR)
    assert os.path.exists(f_n1_0), \
           'Basis set n = {}, degree = {} was not saved!'.format(n1, 0)

    bs_n2_0 = get_bs_cached(n2, 0, basis_dir=DATA_DIR)
    assert os.path.exists(f_n2_0), \
           'Basis set n = {}, degree = {} was not saved!'.format(n2, 0)

    bs_n1_1 = get_bs_cached(n1, 1, basis_dir=DATA_DIR)
    assert os.path.exists(f_n1_1), \
           'Basis set n = {}, degree = {} was not saved!'.format(n1, 1)

    # loading
    cache_cleanup()
    bs = get_bs_cached(n1, 0, basis_dir=DATA_DIR)
    assert_allclose(bs_n1_0, bs, err_msg='Loaded basis set ' +
                    'n = {}, degree = {} differs from saved!'.format(n1, 0))

    cache_cleanup()
    bs = get_bs_cached(n2, 0, basis_dir=DATA_DIR)
    assert_allclose(bs_n2_0, bs, err_msg='Loaded basis set ' +
                    'n = {}, degree = {} differs from saved!'.format(n2, 0))

    cache_cleanup()
    bs = get_bs_cached(n1, 1, basis_dir=DATA_DIR)
    assert_allclose(bs_n1_1, bs, err_msg='Loaded basis set ' +
                    'n = {}, degree = {} differs from saved!'.format(n1, 1))

    # crop-loading
    os.remove(f_n1_0)
    cache_cleanup()
    bs = get_bs_cached(n1, 0, basis_dir=DATA_DIR)
    assert_allclose(bs_n1_0, bs, err_msg='Loaded cropped basis set differs')

    # clean-up
    os.remove(f_n2_0)
    os.remove(f_n1_1)


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

    for degree, tol in [(0, 2e-3), (1, 3e-4), (2, 9e-4), (3, 4e-5)]:
        recon = daun_transform(tr, degree=degree, verbose=False)
        recon = recon[n // 2 + n % 2]
        assert_allclose(recon, ref.func, atol=tol,
                        err_msg='-> degree = ' + str(degree))

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

    for degree, tol in [(0, 3e-2), (1, 7e-3), (2, 2e-2), (3, 7e-4)]:
        proj = daun_transform(tr, degree=degree, direction='forward',
                              verbose=False)
        proj = proj[n // 2 + n % 2]
        assert_allclose(proj, ref.abel, atol=tol,
                        err_msg='-> degree = ' + str(degree))

    # test dr
    proj = daun_transform(tr, dr=0.5, direction='forward', verbose=False)[0]
    assert_allclose(proj, ref.abel / 2, atol=2e-2, err_msg='-> dr = 0.5')


if __name__ == '__main__':
    test_daun_bs()
    test_daun_bs_cache()
    test_daun_bs_disk_cache()
    test_daun_shape()
    test_daun_zeros()
    test_daun_gaussian()
    test_daun_forward_gaussian()
