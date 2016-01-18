# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose
from abel.three_point import iabel_three_point_transform, iabel_three_point, get_bs_three_point_cached
from abel.tools.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark

DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_three_point_basis_sets_cache_asym():
    n = 121
    file_name = os.path.join(DATA_DIR, "three_point_basis_{}_{}.npy".format(n, n))
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    get_bs_three_point_cached(n, basis_dir=DATA_DIR, verbose=False)
    # 2nd call load from file
    get_bs_three_point_cached(n, basis_dir=DATA_DIR, verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)

def test_three_point_step_ratio():
    """Check a gaussian solution for three_point"""

    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=True,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    recon = iabel_three_point(tr, 25, basis_dir=None, verbose=False)
    recon1d = recon[n//2 + n%2]

    ratio = absolute_ratio_benchmark(ref, recon1d)

    assert_allclose( ratio , 1.0, rtol=3e-2, atol=0)
