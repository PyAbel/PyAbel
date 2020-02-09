#!/usr/bin/python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_array_less

import abel
from abel.rbasex import rbasex_transform, cache_cleanup
from abel.tools.analytical import GaussianAnalytical
from abel.hansenlaw import hansenlaw_transform


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


def run_orders(odd=False):
    """
    Test angular orders using Gaussian peaks by comparison with hansenlaw.
    """
    maxorder = 6
    sigma = 5.0  # Gaussian sigma
    step = 6 * sigma  # distance between peak centers

    for order in range(maxorder + 1):
        rmax = int((order + 2) * step)
        if odd:
            if order == 0:
                continue  # 0th order cannot be odd
            height = 2 * rmax + 1
        else:  # even only
            if order % 2:
                continue  # skip odd
            height = rmax + 1
        # coordinates (Q0 or right half):
        x = np.arange(float(rmax + 1))
        y = rmax - np.arange(float(height))[:, None]
        # radius
        r = np.sqrt(x**2 + y**2)
        # cos, sin
        r[rmax, 0] = np.inf
        c = y / r
        s = x / r
        r[rmax, 0] = 0

        # Gaussian peak with one cossin angular term
        def peak(i):
            m = i  # cos power
            k = (order - m) & ~1  # sin power (round down to even)
            return c ** m * s ** k * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # create source distribution
        src = peak(0)
        for i in range(1, order + 1):
            if not odd and i % 2:
                continue  # skip odd
            src += peak(i)

        # reference forward transform
        abel = hansenlaw_transform(src, direction='forward', hold_order=1)

        param = ', order = {}, odd = {}, '.format(order, odd)

        # test forward transform
        for mode in ['clean', 'cached']:
            if mode == 'clean':
                cache_cleanup()
            proj, _ = rbasex_transform(src, origin=(rmax, 0),
                                       order=order, odd=odd,
                                       direction='forward', out='fold')
            assert_allclose(proj, abel, rtol=0.003, atol=0.4,
                            err_msg='-> forward' + param + mode)

        # test inverse transforms
        for reg in [None, ('L2', 1), ('diff', 1), ('SVD', 1 / rmax)]:
            for mode in ['clean', 'cached']:
                if mode == 'clean':
                    cache_cleanup()
                recon, _ = rbasex_transform(abel, origin=(rmax, 0),
                                            order=order, odd=odd,
                                            reg=reg, out='fold')
                recon[rmax-2:rmax+3, :2] = 0  # exclude pixels near center
                assert_allclose(recon, src, atol=0.03,
                                err_msg='-> reg = ' + str(reg) + param + mode)


def test_rbasex_orders():
    run_orders()


def test_rbasex_orders_odd():
    run_orders(odd=True)


def test_rbasex_pos():
    """
    Test positive regularization as in run_orders().
    """
    sigma = 5.0  # Gaussian sigma
    step = 6 * sigma  # distance between peak centers

    for order in [0, 1, 2, 4, 6]:
        rmax = int((order + 2) * step)
        if order == 1:  # odd
            rmax += int(step)  # 3 peaks instead of 2
            height = 2 * rmax + 1
        else:  # even only
            height = rmax + 1
        # coordinates (Q0 or right half):
        x = np.arange(float(rmax + 1))
        y = rmax - np.arange(float(height))[:, None]
        # radius
        r = np.sqrt(x**2 + y**2)
        # cos, sin
        r[rmax, 0] = np.inf
        c = y / r
        s = x / r
        r[rmax, 0] = 0

        # Gaussian peak with one cossin angular term
        def peak(i, isotropic=False):
            if isotropic:
                return np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))
            m = i  # cos power
            k = (order - m) & ~1  # sin power (round down to even)
            return c ** m * s ** k * \
                   np.exp(-(r - (i + 1) * step) ** 2 / (2 * sigma**2))

        # create source distribution
        src = peak(0)
        if order == 1:  # special case
            src += (1 + c) * peak(1, True)  # 1 + cos >= 0
            src += (1 - c) * peak(2, True)  # 1 - cos >= 0
        else:  # other even orders
            for i in range(2, order + 1, 2):
                src += peak(i)
        # array for nonnegativity test (with some tolerance)
        zero = np.full_like(src, -1e-15)

        # reference forward transform
        abel = hansenlaw_transform(src, direction='forward', hold_order=1)
        # with some noise
        abel += 0.05 * np.random.RandomState(0).rand(*abel.shape)

        param = '-> order = {}, '.format(order)

        # test inverse transform
        for mode in ['clean', 'cached']:
            if mode == 'clean':
                cache_cleanup()
            recon, _ = rbasex_transform(abel, origin=(rmax, 0),
                                        order=order,
                                        reg='pos', out='fold')
            recon[rmax-3:rmax+4, :2] = 0  # exclude pixels near center
            assert_allclose(recon, src, atol=0.05,
                            err_msg=param + mode)
            # check nonnegativity
            assert_array_less(zero, recon)


if __name__ == '__main__':
    test_rbasex_shape()
    test_rbasex_zeros()
    test_rbasex_gaussian()
    test_rbasex_orders()
    test_rbasex_orders_odd()
    test_rbasex_pos()
