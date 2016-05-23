# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose
import abel
from abel.benchmark import absolute_ratio_benchmark

DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_linbasex_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.linbasex.linbasex_transform_full(x, basis_dir=None)

    assert recon[0].shape == (n+1, n+1)   # NB shape+1


def test_linbasex_forward_dribinski_image():
    """ Check hansenlaw forward/inverse transform
        using BASEX sample image, comparing speed distributions
    """

    # BASEX sample image
    IM = abel.tools.analytical.sample_image(n=1001, name="dribinski")

    # forward Abel transform
    fIM = abel.Transform(IM, method='hansenlaw', direction='forward')

    # inverse Abel transform
    ifIM = abel.Transform(fIM.transform, method='linbasex',
                          transform_options=dict(sig_s=0, inc=1, 
                                            un=[0, 2], an=[0, 90],
                                            return_Beta=True))

    # speed distribution
    orig_radial, orig_speed = abel.tools.vmi.angular_integration(IM)


    radial = ifIM.radial
    speed = ifIM.Beta[0]

    orig_speed /= orig_speed[1:60].max()
    speed /= speed[1:60].max()

    assert np.allclose(orig_speed[1:60], speed[1:60], rtol=0, atol=0.1)


if __name__ == "__main__":
    test_linbasex_shape()
    test_linbasex_forward_dribinski_image()
