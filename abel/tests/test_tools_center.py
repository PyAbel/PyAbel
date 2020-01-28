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
    # image width, height n = 361, origin = (180, 180)
    IM = abel.tools.analytical.SampleImage(n=361, name="dribinski").image

    # artificially displace origin, now at (179, 182)
    IMx = shift(IM, (-1, 2))
    true_origin = (179, 182)

    # find_origin using 'slice' method
    origin = abel.tools.center.find_origin(IMx, method="slice")

    assert_allclose(origin, true_origin, atol=1)

    # find_origin using 'com' method
    origin = abel.tools.center.find_origin(IMx, method="com")

    assert_allclose(origin, true_origin, atol=1)

    # check single axis - vertical
    # center shifted image IMx in the vertical direction only
    IMc = abel.tools.center.center_image(IMx, method="com", axes=1)
    # determine the origin
    origin = abel.tools.center.find_origin(IMc, method="com")

    assert_allclose(origin, (179, 180), atol=1)

    # check single axis - horizontal
    # center shifted image IMx in the horizontal direction only
    IMc = abel.tools.center.center_image(IMx, method="com", axes=0)
    origin = abel.tools.center.find_origin(IMc, method="com")

    assert_allclose(origin, (180, 182), atol=1)

    # check even image size returns odd
    # drop off one column, to make an even column image
    IM = IM[:, :-1]
    m, n = IM.shape

    IMy = abel.tools.center.center_image(IM, method="slice", odd_size=True)

    assert_allclose(IMy.shape, (m, n-1))


if __name__ == "__main__":
    test_center_image()
