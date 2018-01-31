from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import abel


def test_circularize_image():

    IM = abel.tools.analytical.SampleImage(n=511, name='Ominus', sigma=2).image

    # flower image distortion
    def flower_scaling(theta, freq=2, amp=0.1):
        return 1 + amp*np.sin(freq*theta)**4

    IMdist = abel.tools.circularize.circularize(IM,
                                    radial_correction_function=flower_scaling)

    IMcirc, angle, scalefactor, spline =\
        abel.tools.circularize.circularize_image(IMdist,
                   method='lsq', dr=0.5, dt=0.1, smooth=0,
                   ref_angle=0, return_correction=True)

    r, c = IMcirc.shape

    diff = (IMcirc - IM).sum(axis=1).sum(axis=0)

    assert_allclose(diff, -307.603, atol=0.05)

    assert_allclose(angle[-1], 3.04, atol=0.05)

    assert_allclose(scalefactor[4], 0.97, atol=0.05)


if __name__ == "__main__":
    test_circularize_image()
