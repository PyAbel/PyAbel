from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel

from scipy.ndimage.interpolation import shift


def test_center_image():

    # BASEX sample image, Gaussians at 10, 15, 20, 70,85, 100, 145, 150, 155
    # image width, height n = 361, center = (180, 180)
    IM = abel.tools.analytical.SampleImage(n=361, name="dribinski").image

    # artificially displace center, now at (179, 182)
    IMx = shift(IM, (-1, 2))
    true_center = (179, 182)

    # find_center using 'slice' method
    center = abel.tools.center.find_center(IMx, center="slice")

    assert_allclose(center, true_center, atol=1)

    # find_center using 'com' method
    center = abel.tools.center.find_center(IMx, center="com")

    assert_allclose(center, true_center, atol=1)

    # check single axis - vertical
    # center shifted image IMx in the vertical direction only
    IMc = abel.tools.center.center_image(IMx, center="com", axes=1)
    # determine the center
    center = abel.tools.center.find_center(IMc, center="com")

    assert_allclose(center, (179, 180), atol=1)

    # check single axis - horizontal
    # center shifted image IMx in the horizontal direction only
    IMc = abel.tools.center.center_image(IMx, center="com", axes=0)
    center = abel.tools.center.find_center(IMc, center="com")

    assert_allclose(center, (180, 182), atol=1)

    # check even image size returns odd
    # drop off one column, to make an even column image
    IM = IM[:, :-1]
    m, n = IM.shape

    IMy = abel.tools.center.center_image(IM, center="slice", odd_size=True)

    assert_allclose(IMy.shape, (m, n-1))


if __name__ == "__main__":
    test_center_image()
