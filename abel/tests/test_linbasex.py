# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import abel
from abel.linbasex import linbasex_transform_full


def test_linbasex_shape():
    """ Check output shape with default parameters
    """
    R = 10
    n = 2 * R + 1
    x = np.ones((n, n), dtype='float32')

    recon, radial, beta, proj = linbasex_transform_full(x)

    assert recon.shape == (n, n)
    assert radial.shape == (R + 1,)
    assert beta.shape == (2, R + 1)
    assert proj.shape == (2, n)


def test_linbasex_shape_radial_step():
    """ Check output shape with sparse basis (radial step > 1)
    """
    R = 10
    dR = 2
    n = 2 * R + 1
    x = np.ones((n, n), dtype='float32')

    recon, radial, beta, proj = linbasex_transform_full(x, radial_step=dR)

    assert recon.shape == (n, n)
    assert radial.shape == (R // dR + 1,)
    assert beta.shape == (2, R // dR + 1)
    assert proj.shape == (2, n)


def test_linbasex_shape_clip():
    """ Check output shape with excluded center (clip > 0)
    """
    R = 10
    clip = 3
    n = 2 * R + 1
    x = np.ones((n, n), dtype='float32')

    recon, radial, beta, proj = linbasex_transform_full(x, clip=clip)

    assert recon.shape == (n, n)
    assert radial.shape == (R + 1 - clip,)
    assert radial[0] == clip
    assert radial[-1] == R
    assert beta.shape == (2, R + 1 - clip)
    assert proj.shape == (2, n)


def test_linbasex_zeros():
    """ Check that zero input produces zero output
    """
    R = 10
    n = 2 * R + 1
    x = np.zeros((n, n), dtype='float32')

    recon, radial, beta, proj = linbasex_transform_full(x)

    assert_allclose(recon, 0)
    assert_allclose(beta, 0)
    assert_allclose(proj, 0)


def test_linbasex_forward_dribinski_image():
    """ Check hansenlaw forward/inverse transform
        using BASEX sample image, comparing speed distributions
    """

    # BASEX sample image
    IM = abel.tools.analytical.SampleImage(n=1001, name="dribinski").func

    # forward Abel transform
    fIM = abel.Transform(IM, method='hansenlaw', direction='forward')

    # inverse Abel transform
    ifIM = abel.Transform(fIM.transform, method='linbasex',
                          transform_options=dict(legendre_orders=[0, 2],
                                                 proj_angles=[0, np.pi/2]))

    # speed distribution
    orig_radial, orig_speed = abel.tools.vmi.angular_integration_3D(IM)

    radial = ifIM.radial
    speed = ifIM.Beta[0]

    orig_speed /= orig_speed[1:60].max()
    speed /= speed[1:60].max()

    assert_allclose(orig_speed[1:60], speed[1:60], rtol=0, atol=0.1)


def test_linbasex_odd_sign():
    """Check for sign and output image orientation with odd Legendre order
    """
    R = 10
    x = np.arange(-R, R + 1)
    y = x[:, None]
    r = np.sqrt(x**2 + y**2 + 0.1)  # + 0.1 to avoid division by zero
    # gaussian(r) × (1 + cos θ):
    im = np.exp(-(r / R * 2)**2) * (1 - y / r)
    assert np.all(im[0] > im[-1])  # upper part of cos θ is positive

    recon, radial, beta, proj = linbasex_transform_full(im, legendre_orders=[0, 1])

    assert np.all(recon[0] > recon[-1]), 'incorrect output orientation'
    assert_array_less(-1e-9, beta[0][1:], err_msg='beta[0] must be positive')
    assert_array_less(-1e-9, beta[1][1:], err_msg='beta[1] must be positive')


def test_linbasex_mean_beta():
    """Check integrated intensities and averaged anisotropies using the
       Lin-BASEX test image
    """
    im = abel.tools.analytical.SampleImage(n=513, name='Gerber', sigma=5).abel

    # original radii and beta values for Gerber sample image:
    # sphere: 1     2    3    4    5    6      7     8
    r_ref = [38,   70,  90,  134, 138, 143,   196,  230 ]
    beta_ref = np.array([
            [ 1.2,  1.5, 1.5, 2,   1.8, 1,     2,    2  ],   # β₀
            [-0.4, -1,   1,   1,   0.5, 0.5,   1,    1  ],   # β₁
            [ 0,    0.5, 0.4, 0.4, 0,   0.25, -0.5, -0.5]])  # β₂
    # combine overlapping spheres 4–6 (indices 3–5)
    dr = 10
    regions = [(r - dr, r + dr) for r in r_ref[:3]] + \
              [(r_ref[3] - dr, r_ref[5] + dr)] + \
              [(r - dr, r + dr) for r in r_ref[6:]]
    ref = np.empty((3, 6))
    ref[:, :3] = beta_ref[:, :3]
    ref[:, 3] = beta_ref[:, 3:6].sum(axis=1)
    ref[:, 4:] = beta_ref[:, 6:]
    # normalize to β₀
    ref[1:] /= ref[0]
    # scale overall intensity
    ref[0] *= 2.6

    def check(I_tol, beta_tol, radial_step=1, clip=0):
        recon, radial, beta, proj = linbasex_transform_full(
            im,
            proj_angles=[0, np.pi/4, np.pi/2],
            legendre_orders=[0, 2, 4],
            radial_step=radial_step,
            threshold=0.01,
            clip=clip
        )

        beta_mean = abel.linbasex.mean_beta(radial, beta, regions)

        param = 'radial_step={}, clip={}'.format(radial_step, clip)
        assert_allclose(beta_mean[0], ref[0], rtol=I_tol,
                        err_msg=param + ': I')
        assert_allclose(beta_mean[1:], ref[1:], atol=beta_tol,
                        err_msg=param + ': beta')

    # default parameters:
    check(0.02, 0.03)
    # sparse, without center:
    check(0.03, 0.03, radial_step=2, clip=10)


if __name__ == "__main__":
    test_linbasex_shape()
    test_linbasex_shape_radial_step()
    test_linbasex_shape_clip()
    test_linbasex_zeros()
    test_linbasex_forward_dribinski_image()
    test_linbasex_odd_sign()
    test_linbasex_mean_beta()
