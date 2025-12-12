import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest

import abel
from abel import direct
from abel.tools.analytical import GaussianAnalytical

def test_direct_shape():
    """Ensure that abel.direct.direct_transform() returns an array of the correct shape"""

    n = 21
    x = np.ones((n, n))

    recon = direct.direct_transform(x, direction='forward')
    assert recon.shape == (n, n)

    recon = direct.direct_transform(x, direction='inverse')
    assert recon.shape == (n, n)


def test_direct_zeros():
    """Test abel.direct.direct_transform() with zeros"""
    n = 64
    x = np.zeros((n, n))
    assert_equal(direct.direct_transform(x, direction='forward'), 0)
    assert_equal(direct.direct_transform(x, direction='inverse'), 0)


def test_direct_gaussian():
    """Check abel.direct.direct_transform() against the analytical Gaussian"""

    n = 501
    r_max = 100

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=20)

    # forward:
    recon = direct.direct_transform(ref.func, dr=ref.dr, direction='forward')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='forward')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)

    # inverse:
    recon = direct.direct_transform(ref.abel, dr=ref.dr, direction='inverse')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='inverse')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not direct.cython_ext,
                    reason='abel.direct C extension not installed')
def test_direct_c_python_correspondence_with_correction():
    """ Check that both the C and Python backends are identical (correction=True)"""
    N = 10
    r = 0.5 + np.arange(N).astype('float64')
    x = 2*r.reshape((1, -1))**2
    out1 = direct._pyabel_direct_integral(x, r, 1)
    out2 = direct._cabel_direct_integral(x, r, 1)
    assert_allclose(out1, out2, rtol=1e-9, atol=1e-9)


@pytest.mark.skipif(not direct.cython_ext,
                    reason='abel.direct C extension not installed')
def test_direct_c_python_correspondence():
    """ Check that both the C and Python backends are identical (correction=False)"""
    N = 10
    r = 0.5 + np.arange(N).astype('float64')
    x = 2*r.reshape((1, -1))**2

    out1 = direct._pyabel_direct_integral(x, r, 0)
    out2 = direct._cabel_direct_integral( x, r, 0)
    assert_allclose(out1, out2, rtol=1e-9, atol=1e-9)


def test_direct_new():
    """ Check that new implementation reproduces old results """
    n = 51
    r_max = 100
    n_min = 10

    gauss = GaussianAnalytical(n, r_max, symmetric=False, sigma=20)
    r_cut = gauss.r[n_min:]

    opt = {'dr': gauss.dr, 'backend': 'Python'}
    opt_cut = {'r': r_cut, 'backend': 'Python'}
    fwd = {'direction': 'forward'}
    nocor = {'correction': False}

    # inverse:
    im = np.vstack((gauss.abel, [0] * n, [1] * n))
    res = direct.direct_transform(im, **opt)
    res_new = direct.direct_transform_new(im, **opt)
    assert_allclose(res_new, res, err_msg='-> inverse')
    res_cut = direct.direct_transform_new(im[:, n_min:], **opt_cut)
    # (1st point is slightly off due to right/central derivative discrepancy)
    assert_allclose(res_cut[:, 0], res[:, n_min], rtol=2e-2,
                    err_msg='-> inverse cut [0]')
    assert_allclose(res_cut[:, 1:], res[:, n_min+1:], err_msg='-> inverse cut')
    # without correction:
    res = direct.direct_transform(im, **opt, **nocor)
    res_new = direct.direct_transform_new(im, **opt, **nocor)
    assert_allclose(res_new, res, err_msg='-> inverse, correction=False')
    res_cut = direct.direct_transform_new(im[:, n_min:], **opt_cut, **nocor)
    assert_allclose(res_cut, res[:, n_min:],
                    err_msg='-> inverse cut, correction=False')

    # forward:
    im = np.vstack((gauss.func, [0] * n, [1] * n))
    res = direct.direct_transform(im, **opt, **fwd)
    res_new = direct.direct_transform_new(im, **opt, **fwd)
    assert_allclose(res_new, res, err_msg='-> forward')
    res_cut = direct.direct_transform_new(im[:, n_min:], **opt_cut, **fwd)
    assert_allclose(res_cut, res[:, n_min:], err_msg='-> forward cut')
    # without correction:
    res = direct.direct_transform(im, **opt, **fwd, **nocor)
    res_new = direct.direct_transform_new(im, **opt, **fwd, **nocor)
    assert_allclose(res_new, res, err_msg='-> forward, correction=False')
    res_cut = direct.direct_transform_new(im[:, n_min:], **opt_cut, **fwd,
                                          **nocor)
    assert_allclose(res_cut, res[:, n_min:],
                    err_msg='-> forward cut, correction=False')


@pytest.mark.skipif(not direct.cython_ext,
                    reason='abel.direct C extension not installed')
def test_direct_c_new_with_correction():
    """ Check that new and old C backends are identical (correction=True) """
    N = 10
    Nc = 1
    r = np.arange(N, dtype=float)
    x = 2 * np.atleast_2d(r)**2

    res = direct._cabel_direct_integral(x, r, 1)
    res_new = direct._cabel_direct_integral_new(x, r, 1)
    assert_allclose(res_new, res)
    res_cut = direct._cabel_direct_integral_new(x[:, Nc:], r[Nc:], 1)
    assert_allclose(res_cut, res_new[:, Nc:])


@pytest.mark.skipif(not direct.cython_ext,
                    reason='abel.direct C extension not installed')
def test_direct_c_new():
    """ Check that new and old C backends are identical (correction=False) """
    N = 10
    Nc = 1
    r = np.arange(N, dtype=float)
    x = 2 * np.atleast_2d(r)**2

    res = direct._cabel_direct_integral(x, r, 0)
    res_new = direct._cabel_direct_integral_new(x, r, 0)
    assert_allclose(res_new, res)
    res_cut = direct._cabel_direct_integral_new(x[:, Nc:], r[Nc:], 0)
    assert_allclose(res_cut, res_new[:, Nc:])


if __name__ == "__main__":
    test_direct_shape()
    test_direct_zeros()
    test_direct_gaussian()
    test_direct_c_python_correspondence_with_correction()
    test_direct_c_python_correspondence()

    test_direct_new()
    test_direct_c_new_with_correction()
    test_direct_c_new()
