# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose, assert_array_less
import abel
from abel.linbasex import linbasex_transform_full


def test_linbasex_shape():
    R = 10
    n = 2 * R + 1
    x = np.ones((n, n), dtype='float32')

    recon, radial, beta, proj = linbasex_transform_full(x)

    assert recon.shape == (n, n)
    assert radial.shape == (R + 1,)
    assert beta.shape == (2, R + 1)
    assert proj.shape == (2, n)


def test_linbasex_shape_radial_step():
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
    R = 10
    x = np.arange(-R, R + 1)
    y = x[:, None]
    r = np.sqrt(x**2 + y**2 + 0.1)  # + 0.1 to avoid division by zero
    # gaussian(r) × (1 + cos θ):
    im = np.exp(-(r / R * 2)**2) * (1 - y / r)
    assert np.all(im[0] > im[-1])  # upper part of cos θ is positive

    recon, radial, beta, proj = linbasex_transform_full(im, legendre_orders=[0, 1])

    assert np.all(recon[0] > recon[-1]), 'incorrect output orientation'
    # ignoring r = 0:
    assert_array_less(0, beta[0][1:], err_msg='beta[0] must be positive')
    assert_array_less(0, beta[1][1:], err_msg='beta[1] must be positive')


if __name__ == "__main__":
    test_linbasex_shape()
    test_linbasex_shape_radial_step()
    test_linbasex_shape_clip()
    test_linbasex_forward_dribinski_image()
    test_linbasex_odd_sign()
