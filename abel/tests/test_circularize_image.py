from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import abel
import scipy.interpolate


def test_circularize_image():

    IM = abel.tools.analytical.sample_image(n=511, name='Ominus', sigma=2)

    IMdist = abel.tools.analytical.flower_distort(IM)

    nslices = 32

    IMcirc, angle, scalefactor, spline =\
        abel.tools.circularize.circularize_image(IMdist,
                   method='lsq', nslices=nslices, zoom=1, smooth=0,
                   return_correction=True)

    r, c = IMcirc.shape

    diff = (IMcirc - IM).sum(axis=1).sum(axis=0)

    assert int(diff) == 1542426

    assert len(angle) == nslices

    assert int(angle[-1]*100) == 304
    
    assert int(scalefactor[4]*100) == 92


if __name__ == "__main__":
    test_circularize_image()
