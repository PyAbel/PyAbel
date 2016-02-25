from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_basex_basis_sets_cache():
    # n_vert, n_horz = 121,121
    # nbf_vert, nbf_horz = 121, 61
    n = 121
    file_name = os.path.join(DATA_DIR, "basex_basis_{}_{}_{}_{}.npy".format(n, n, n, n//2+1))
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    abel.basex.get_bs_basex_cached(n,n, basis_dir=DATA_DIR, verbose=False)
    # 2nd call load from file
    abel.basex.get_bs_basex_cached(n,n, basis_dir=DATA_DIR, verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)


def test_basex_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')
    bs = abel.basex.get_bs_basex_cached(n,n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(x, *bs)

    assert recon.shape == (n, n) 

def test_basex_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')
    bs = abel.basex.get_bs_basex_cached(n,n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(x, *bs)

    assert_allclose(recon, 0)


def test_basex_step_ratio():
    """Check a gaussian solution for BASEX"""
    n = 51
    r_max = 25

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, symmetric=True,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    bs = abel.basex.get_bs_basex_cached(n,n, basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(tr, *bs)
    recon1d = recon[n//2 + n%2]

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)

    assert_allclose( ratio , 1.0, rtol=3e-2, atol=0)

if __name__ == '__main__':
    test_basex_basis_sets_cache()
    test_basex_shape()
    test_basex_zeros()
    test_basex_step_ratio()
