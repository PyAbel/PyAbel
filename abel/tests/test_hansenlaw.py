from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import abel


def test_hansenlaw_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction='inverse')

    assert recon.shape == (n, n)


def test_hansenlaw_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction="inverse")

    assert_allclose(recon, 0)


def test_hansenlaw_forward_tansform_gaussian():
    """Check hansenlaw forward tansform with a Gaussian function"""
    n = 1001
    r_max = 501   # more points better fit

    ref = abel.tools.analytical.GaussianAnalytical(n,
                     r_max, symmetric=False,  sigma=200)

    recon = abel.hansenlaw.hansenlaw_transform(ref.func, ref.dr,
                                               direction='forward')

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='forward')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)


def test_hansenlaw_inverse_transform_gaussian():
    """Check hansenlaw inverse transform with a Gaussian function"""
    n = 1001   # better with a larger number of points
    r_max = 501

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max,
                     symmetric=False,  sigma=200)
    tr = np.tile(ref.abel[None, :], (n, 1))  # make a 2D array from 1D

    recon = abel.hansenlaw.hansenlaw_transform(tr, ref.dr, direction='inverse')
    recon1d = recon[n//2]  # center row

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)

    assert_allclose(ratio, 1.0, rtol=1e-1, atol=0)


def test_hansenlaw_forward_curveA():
    """ Check hansenlaw forward transform with 'curve A'
    """

    n = 101
    curveA = abel.tools.analytical.TransformPair(n, profile=3)

    # forward Abel == g(r)
    Aproj = abel.hansenlaw.hansenlaw_transform(curveA.func, curveA.dr,
                                               direction='forward')


    assert_allclose(curveA.abel, Aproj, rtol=0, atol=8.0e-2)


def test_hansenlaw_inverse_transform_curveA():
    """ Check hansenlaw inverse transform() 'curve A'
    """

    n = 101
    curveA = abel.tools.analytical.TransformPair(n, profile=3)

    # inverse Abel == f(r)
    recon = abel.hansenlaw.hansenlaw_transform(curveA.abel, curveA.dr,
                                               direction='inverse')

    assert_allclose(curveA.func[:n//2], recon[:n//2], rtol=0.09, atol=0)


def test_hansenlaw_forward_dribinski_image():
    """ Check hansenlaw forward/inverse transform
        using BASEX sample image, comparing speed distributions
    """

    # BASEX sample image
    IM = abel.tools.analytical.SampleImage(n=1001, name="dribinski").image

    # core transform(s) use top-right quadrant, Q0
    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(IM)

    # forward Abel transform
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    # inverse Abel transform
    ifQ0 = abel.hansenlaw.hansenlaw_transform(fQ0, direction='inverse')

    # speed distribution
    orig_speed, orig_radial = abel.tools.vmi.angular_integration(Q0,
                              origin=(0, 0), Jacobian=True)

    speed, radial_coords = abel.tools.vmi.angular_integration(ifQ0,
                           origin=(0, 0), Jacobian=True)

    orig_speed /= orig_speed[50:125].max()
    speed /= speed[50:125].max()

    assert np.allclose(orig_speed[50:125], speed[50:125], rtol=0.5, atol=0)

if __name__ == "__main__":
    test_hansenlaw_shape()
    test_hansenlaw_zeros()
    test_hansenlaw_forward_tansform_gaussian()
    test_hansenlaw_inverse_transform_gaussian()
    test_hansenlaw_forward_curveA()
    test_hansenlaw_inverse_transform_curveA()
    test_hansenlaw_forward_dribinski_image()
