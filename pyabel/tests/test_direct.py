import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pytest

import abel
from abel import direct
from abel.direct import direct_transform
from abel.tools.analytical import GaussianAnalytical, TransformPair
from abel.tools.math import trapezoid


def test_direct_shape():
    """Ensure that direct_transform() returns an array of the correct shape"""

    n = 21
    x = np.ones((n, n))

    recon = direct_transform(x, direction='forward')
    assert recon.shape == (n, n)

    recon = direct_transform(x, direction='inverse')
    assert recon.shape == (n, n)


def test_direct_zeros():
    """Test direct_transform() with zeros"""
    n = 64
    x = np.zeros((n, n))
    assert_equal(direct_transform(x, direction='forward'), 0)
    assert_equal(direct_transform(x, direction='inverse'), 0)


def test_direct_gaussian():
    """Check direct_transform() against the analytical Gaussian"""

    n = 501
    r_max = 100

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=20)

    # forward:
    recon = direct_transform(ref.func, dr=ref.dr, direction='forward')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='forward')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)

    # inverse:
    recon = direct_transform(ref.abel, dr=ref.dr, direction='inverse')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='inverse')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)


def test_direct_preserve():
    """Check that direct_transform() does not corrupt the input image"""
    n = 10
    IM = np.ones((n, n), dtype=float)
    IM_copy = IM.copy()
    for direction in ['forward', 'inverse']:
        for background in [None, 0]:
            direct_transform(IM, direction=direction, background=background)
            assert_equal(IM, IM_copy, err_msg='-> IM corrupted: '
                         f'{direction=}, {background=}')


def test_direct_background():
    """Test direct_transform() with various background modes"""
    n = 10
    # profiles with the step at last pixel's:
    pairs = (TransformPair(n, 8),  # inner edge
             TransformPair(n, 5),  # center
             TransformPair(n, 9))  # outer edge
    funcs = np.vstack([p.func for p in pairs])
    funcs = np.vstack([funcs, funcs + 1])  # + shifted intensity
    abels = np.vstack([p.abel for p in pairs])
    abels = np.vstack([abels, abels + 1])  # + shifted intensity
    dr = pairs[0].dr

    # Forward transforms:
    kwargs = {'dr': dr, 'direction': 'forward'}
    fwd_None = direct_transform(funcs, **kwargs, background=None)
    # "inner edge" OK
    assert_allclose(fwd_None[0], abels[0], atol=5e-2)
    # "center" OK
    assert_allclose(fwd_None[1], abels[1], atol=2e-2)
    # "outer edge" = "center"
    assert_allclose(fwd_None[2], fwd_None[1])
    # "center" + 1 = "center" × 2
    assert_allclose(fwd_None[4], fwd_None[1] * 2)

    fwd_0 = direct_transform(funcs, **kwargs, background=0)
    # "inner edge" OK
    assert_allclose(fwd_0[0], abels[0], atol=5e-2)
    # "outer edge" OK
    assert_allclose(fwd_0[2], abels[2], atol=6e-2)
    # "center" = "outer edge"
    assert_allclose(fwd_0[1], fwd_0[2])
    # "outer edge" + 1 = "outer edge" × 2
    assert_allclose(fwd_0[5], fwd_0[2] * 2)

    fwd_1 = direct_transform(funcs, **kwargs, background=1)
    # "center" extended by background OK
    abel_p5ext = 2 * np.sqrt((1 + dr)**2 - pairs[1].r**2)
    assert_allclose(fwd_1[1], abel_p5ext, atol=2e-2)

    # Inverse transforms:
    kwargs['direction'] = 'inverse'
    inv_None = direct_transform(abels, **kwargs, background=None)
    # insensitive to shift
    assert_allclose(inv_None[:3], inv_None[3:], atol=1e-9)
    # last pixel = 0
    assert_allclose(inv_None[:, -1], 0)
    # "inner edge" OK
    assert_allclose(inv_None[0, :-2], 1, atol=2e-2)
    # "outer edge" not OK
    assert_raises(AssertionError,
                  assert_allclose, inv_None[2, :-2], 1, atol=2e-2)

    inv_0 = direct_transform(abels, **kwargs, background=0)
    # sensitive to shift
    assert_raises(AssertionError,
                  assert_allclose, inv_0[:3], inv_0[3:], atol=1e-9)
    # last pixel ≠ 0
    assert_raises(AssertionError,
                  assert_allclose, inv_0[:, -1], 0)
    # "inner edge" OK
    assert_allclose(inv_0[0, :-2], 1, atol=2e-2)
    # "outer edge" OK
    assert_allclose(inv_0[2, :-1], 1, atol=2e-2)

    inv_1 = direct_transform(abels, **kwargs, background=1)
    # shifted minus background = original with zero background
    assert_allclose(inv_1[3:], inv_0[:3])


@pytest.mark.skipif(not direct.cython_ext,
                    reason='abel.direct C extension not installed')
def test_direct_c_python_correspondence():
    """ Check that both the C and Python backends are identical """
    N = 10
    r = 0.5 + np.arange(N).astype('float64')
    x = 2*r.reshape((1, -1))**2

    for correction in [0, 1]:
        out1 = abel.direct._pyabel_direct_integral(x, r, correction, trapezoid)
        out2 = abel.direct._cabel_direct_integral(x, r, correction)
        assert_allclose(out1, out2, rtol=1e-9, atol=1e-9,
                        err_msg=f'-> {correction=}')


if __name__ == "__main__":
    test_direct_shape()
    test_direct_zeros()
    test_direct_gaussian()
    test_direct_preserve()
    test_direct_background()
    test_direct_c_python_correspondence()
