from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import abel
from abel.rbasex import rbasex_transform
from abel.tools.analytical import GaussianAnalytical


def test_rbasex_shape():
    rmax = 11
    n = 2 * rmax - 1
    im = np.ones((n, n), dtype='float')

    fwd_im, fwd_distr = rbasex_transform(im, direction='forward')
    assert fwd_im.shape == (n, n)
    assert fwd_distr.r.shape == (rmax,)

    inv_im, inv_distr = rbasex_transform(im)
    assert inv_im.shape == (n, n)
    assert inv_distr.r.shape == (rmax,)


def test_rbasex_zeros():
    rmax = 11
    n = 2 * rmax - 1
    im = np.zeros((n, n), dtype='float')

    fwd_im, fwd_distr = rbasex_transform(im, direction='forward')
    assert fwd_im.shape == (n, n)
    assert fwd_distr.r.shape == (rmax,)

    inv_im, inv_distr = rbasex_transform(im)
    assert inv_im.shape == (n, n)
    assert inv_distr.r.shape == (rmax,)


def test_rbasex_gaussian():
    """Check an isotropic gaussian solution for rBasex"""
    rmax = 100
    sigma = 30
    n = 2 * rmax - 1

    ref = GaussianAnalytical(n, rmax, symmetric=True, sigma=sigma)
    # images as direct products
    src = ref.func * ref.func[:, None]
    proj = ref.abel * ref.func[:, None]  # (vertical is not Abel-transformed)

    fwd_im, fwd_distr = rbasex_transform(src, direction='forward')
    # whole image
    assert_allclose(fwd_im, proj, rtol=0.02, atol=0.001)
    # radial intensity profile (without r = 0)
    assert_allclose(fwd_distr.harmonics()[0, 1:], ref.abel[rmax:],
                    rtol=0.02, atol=5e-4)

    inv_im, inv_distr = rbasex_transform(proj)
    # whole image
    assert_allclose(inv_im, src, rtol=0.02, atol=0.02)
    # radial intensity profile (without r = 0)
    assert_allclose(inv_distr.harmonics()[0, 1:], ref.func[rmax:],
                    rtol=0.02, atol=1e-4)


if __name__ == '__main__':
    test_rbasex_shape()
    test_rbasex_zeros()
    test_rbasex_gaussian()
