# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
from numpy.testing import assert_allclose
from abel.three_point import iabel_three_point_transform, OP_D, OP0, OP1, iabel_three_point
from abel.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark

def test_three_point_step_ratio():
    """Check a gaussian solution for three_point"""

    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=True,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    recon = iabel_three_point(tr, 25)
    recon1d = recon[n//2 + n%2]

    ratio = absolute_ratio_benchmark(ref, recon1d)

    assert_allclose( ratio , 1.0, rtol=3e-2, atol=0)
