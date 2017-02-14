from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_equal, assert_almost_equal

import abel


def test_circularize_image():

    IM = abel.tools.analytical.sample_image(n=511, name='Ominus', sigma=2)

    # flower image distortion
    def flower_scaling(theta, freq=2, amp=0.1):
        return 1 + amp*np.sin(freq*theta)**4

    IMdist = abel.tools.circularize.circularize(IM, radcorrspl=flower_scaling)

    nslices = 32

    IMcirc, angle, scalefactor, spline =\
        abel.tools.circularize.circularize_image(IMdist,
                   method='lsq', nslices=nslices, zoom=1, smooth=0,
                   ref_angle=0, return_correction=True)

    r, c = IMcirc.shape

    diff = (IMcirc - IM).sum(axis=1).sum(axis=0)

    assert_almost_equal(diff, -283.7, decimal=0)

    assert_equal(len(angle), nslices)

    assert_almost_equal(angle[-1], 3.04, decimal=2)

    assert_almost_equal(scalefactor[4], 0.92, decimal=2)


if __name__ == "__main__":
    test_circularize_image()
