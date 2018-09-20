from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_basex_basis_sets_cache():
    # n = 61  (121 full width)
    # nbf = 61
    n = 61
    file_name = os.path.join(DATA_DIR, "basex_basis_{}_{}.npy".format(n, n))
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    abel.basex.get_bs_basex_cached(n, basis_dir=DATA_DIR, verbose=False)
    # 2nd call load from file
    abel.basex.get_bs_basex_cached(n, basis_dir=DATA_DIR, verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)


def test_basex_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')
    Ai = abel.basex.get_bs_basex_cached(n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(x, Ai)

    assert recon.shape == (n, n) 

def test_basex_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')
    Ai = abel.basex.get_bs_basex_cached(n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(x, Ai)

    assert_allclose(recon, 0)


def test_basex_step_ratio():
    """Check a gaussian solution for BASEX"""
    n = 26
    r_max = 25

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, symmetric=False, sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    Ai = abel.basex.get_bs_basex_cached(n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(tr, Ai)
    recon1d = recon[n//2 + n%2]

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)

    assert_allclose( ratio , 1.0, rtol=3e-2, atol=0)

if __name__ == '__main__':
    test_basex_basis_sets_cache()
    test_basex_shape()
    test_basex_zeros()
    test_basex_step_ratio()
