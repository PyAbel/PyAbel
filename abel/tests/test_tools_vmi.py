from __future__ import absolute_import
from __future__ import division

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from abel.tools.analytical import SampleImage
from abel.tools.vmi import angular_integration, average_radial_intensity,\
                           radial_integration, anisotropy_parameter, toPES

# The Distributions class and related functions are tested separately in
# test_tools_distributions.py


def make_images(R):
    """
    Make test images (square, centered): 1, cos^2, sin^2
    """
    n = 2 * R + 1  # image size
    ones = np.ones((n, n))

    x = np.arange(float(n)) - R
    x2 = x**2
    r2 = x2 + x2[:, None]
    r2[R, R] = np.inf  # for sin = 0 at r = 0

    sin2 = x2 / r2
    cos2 = 1 - sin2

    return n, ones, cos2, sin2


def test_angular_integration():
    """
    Basic tests of angular integration
    (in its current form, with wrong output).
    """
    R = 50  # image radius
    n, ones, cos2, sin2 = make_images(R)

    def check(name, ref, rtol, IM, **kwargs):
        r, speeds = angular_integration(IM, **kwargs)
        # (ignoring pixels at r = 0 and 1, which can be poor)
        assert_allclose(speeds[2:R], ref(r[2:R]), rtol=rtol,
                        err_msg='-> {}, {}'.format(name, kwargs))

    # for Jacobian=True
    def spher(r):
        return 4 * r**2  # (in fact, must be 4 * np.pi * r**2)

    check('ones', spher, 0.001, ones)
    check('ones', spher, 0.001, ones, dr=0.5)
    check('ones', spher, 0.0001, ones, dt=0.01)
    check('cos2', spher, 0.01, cos2 * 3)
    check('sin2', spher, 0.01, sin2 * 3/2)

    # for Jacobian=False
    def polar(r):
        return np.full_like(r, 2 * np.pi)  # (in fact, should be 2 * np.pi * r)

    check('ones', polar, 0.02, ones, Jacobian=False)
    check('ones', polar, 0.02, ones, Jacobian=False, dr=0.5)
    check('ones', polar, 0.01, ones, Jacobian=False, dt=0.01)
    check('cos2', polar, 0.03, cos2 * 2, Jacobian=False)
    check('sin2', polar, 0.001, sin2 * 2, Jacobian=False)


def test_average_radial_intensity():
    """
    Basic tests of angular averaging.
    """
    R = 50  # image radius
    n, ones, cos2, sin2 = make_images(R)

    def check(name, ref, atol, IM, **kwargs):
        r, intensity = average_radial_intensity(IM, **kwargs)
        assert_allclose(intensity[2:R], ref, atol=atol,
                        err_msg='-> {}, {}'.format(name, kwargs))

    check('ones', 1, 0.02, ones)
    check('ones', 1, 0.02, ones, dr=0.5)
    check('ones', 1, 0.01, ones, dt=0.01)
    check('cos2', 1/2, 0.02, cos2)
    check('sin2', 1/2, 0.001, sin2)


def test_anisotropy_parameter():
    """
    Basic tests of anisotropy fitting.
    """
    n = 100
    theta = np.linspace(-np.pi, np.pi, n, endpoint=False)

    ones = np.ones_like(theta)
    cos2 = np.cos(theta)**2
    sin2 = np.sin(theta)**2

    def check(name, ref, theta, intensity):
        beta, amplitude = anisotropy_parameter(theta, intensity)
        assert_allclose((beta[0], amplitude[0]), ref, atol=1e-8,
                        err_msg='-> ' + name)

    check('ones', (0, 1), theta, ones)
    check('cos2', (2, 1/3), theta, cos2)
    check('sin2', (-1, 2/3), theta, sin2)
    check('cos2sin2', (0, 1/8), theta, cos2 * sin2)


def test_radial_integration():
    """
    Basic test of radial integration.
    """
    # test image (not projected)
    IM = SampleImage(name='dribinski').image

    Beta, Amplitude, Rmidpt, _, _ = \
        radial_integration(IM, radial_ranges=([(65, 75), (80, 90), (95, 105)]))

    assert_equal(Rmidpt, [70, 85, 100], err_msg='Rmidpt')
    assert_allclose([b[0] for b in Beta], [0, 1.58, -0.85], atol=0.01,
                    err_msg='Beta')
    assert_allclose([a[0] for a in Amplitude], [880, 600, 560], rtol=0.01,
                    err_msg='Amplitude')


def test_toPES():
    """
    Check that toPES at least does not crash.
    TODO: do some meaningful tests.
    """
    # test image (not projected)
    IM = SampleImage(name='dribinski').image

    # (parameters from example_hansenlaw.py, although for a different image)
    eBE, PES = toPES(*angular_integration(IM),
                     energy_cal_factor=1.204e-5,
                     photon_energy=1.0e7/454.5, Vrep=-2200,
                     zoom=IM.shape[-1]/2048)


if __name__ == "__main__":
    test_angular_integration()
    test_average_radial_intensity()
    test_anisotropy_parameter()
    test_radial_integration()
    test_toPES()
