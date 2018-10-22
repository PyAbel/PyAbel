# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
import abel

dasch_transforms = {
 "two_point": abel.dasch.two_point_transform,
 "three_point": abel.dasch.three_point_transform,
 "onion_peeling": abel.dasch.onion_peeling_transform
}


def test_dasch_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    for method in dasch_transforms.keys():
        recon = dasch_transforms[method](x, direction='inverse')
        np.testing.assert_equal(recon.shape, (n, n))


def test_dasch_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    for method in dasch_transforms.keys():
        recon = dasch_transforms[method](x, direction="inverse")
        np.testing.assert_allclose(recon, 0)


def test_dasch_deconvolution_array_sources():
    im = abel.tools.analytical.SampleImage(101).image
    q = abel.tools.symmetry.get_image_quadrants(im)[0]

    # clean up any old deconvolution array files
    fn = 'three_point_basis_{}.npy'.format(q.shape[0])
    if os.path.exists(fn):
        os.system('rm {}'.format(fn))

    gb = abel.dasch.three_point_transform(q)
    np.testing.assert_equal(abel.dasch._source, 'generated')

    cb = abel.dasch.three_point_transform(q)
    np.testing.assert_equal(abel.dasch._source, 'cache')
    np.testing.assert_allclose(gb, cb)

    abel.dasch.cache_cleanup()
    fb = abel.dasch.three_point_transform(q)
    np.testing.assert_equal(abel.dasch._source, 'file')
    np.testing.assert_allclose(gb, fb)


def test_dasch_1d_gaussian(n=101):
    def gauss(r, r0, sigma):
        return np.exp(-(r-r0)**2/sigma**2)

    rows, cols = n, n
    r2 = rows//2
    c2 = cols//2

    sigma = 20*n/100

    # 1D Gaussian -----------
    r = np.linspace(0, c2-1, c2)

    orig = gauss(r, 0, sigma)

    for method in dasch_transforms.keys():
        orig_copy = orig.copy()

        recon = dasch_transforms[method](orig)

        ratio_1d = np.sqrt(np.pi)*sigma

        np.testing.assert_allclose(orig_copy[20:], recon[20:]*ratio_1d,
                                   rtol=0.0, atol=0.5)


def test_dasch_cyl_gaussian(n=101):
    def gauss(r, r0, sigma):
        return np.exp(-(r-r0)**2/sigma**2)

    image_shape = (n, n)
    rows, cols = image_shape
    r2 = rows//2
    c2 = cols//2
    sigma = 20*n/100

    x = np.linspace(-c2, c2, cols)
    y = np.linspace(-r2, r2, rows)

    X, Y = np.meshgrid(x, y)

    IM = gauss(X, 0, sigma)  # cylindrical Gaussian located at pixel R=0
    Q0 = IM[:r2, c2:]  # quadrant, top-right
    Q0_copy = Q0.copy()
    ospeed = abel.tools.vmi.angular_integration(Q0_copy, origin=(0, 0))

    # dasch method inverse Abel transform
    for method in dasch_transforms.keys():
        Q0_copy = Q0.copy()
        AQ0 = dasch_transforms[method](Q0)
        ratio_2d = np.sqrt(np.pi)*sigma

        np.testing.assert_allclose(Q0_copy, AQ0*ratio_2d, rtol=0.0, atol=0.3)


if __name__ == "__main__":
    test_dasch_shape()
    test_dasch_zeros()
    test_dasch_deconvolution_array_sources()
    test_dasch_1d_gaussian()
    test_dasch_cyl_gaussian()
