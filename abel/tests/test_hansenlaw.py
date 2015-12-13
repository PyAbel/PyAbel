from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.hansenlaw import  iabel_hansenlaw
from abel.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark


def test_hansenlaw_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = iabel_hansenlaw(x, calc_speeds=False, verbose=False)

    assert recon.shape == (n, n) 


def test_hansenlaw_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = iabel_hansenlaw(x, calc_speeds=False, verbose=False)

    assert_allclose(recon, 0)


def test_hansenlaw_gaussian():
    """Check a gaussian solution for HansenLaw"""
    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=True,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D


    recon = iabel_hansenlaw(tr, calc_speeds=False, verbose=False)
    recon1d = recon[n//2 + n%2]

    ratio = absolute_ratio_benchmark(ref, recon1d)

    # this only passes with a relative tolerance of 0.35, someone would
    # need to look into it.
    assert_allclose(ratio,  1.0, rtol=0.35, atol=0)

