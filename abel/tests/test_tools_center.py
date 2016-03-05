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
    # image width, height n = 361
    IM = abel.tools.analytical.sample_image(n=361, name="dribinski")
    
    # artificially displace center
    IMx = shift(IM, (-1,2))

    # find_center using 'slice' method
    center = abel.tools.center.find_center(IMx, center="slice", axis=(0,1)) 

    assert np.allclose(center, (179, 182), rtol=0, atol=0.1)

    # find_center using 'com' method
    center = abel.tools.center.find_center(IMx, center="com")
   
    assert np.allclose(center, (179, 182), rtol=0, atol=0.4)

    # check even image size returns odd
    IM = IM[:, :-1]
    m, n = IM.shape

    IMy = abel.tools.center.center_image(IM, center="slice", odd_size=True)

    assert IMy.shape == (m, n-1)


if __name__ == "__main__":
  test_center_image()
