from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.basex import BASEX
from abel.io import basex_parse_matlab_basis_sets
from abel.basex import basex_generate_basis_sets, basex_get_basis_sets_cached
from abel.analytical import StepAnalytical
from abel.benchmark import absolute_ratio_benchmark


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def assert_equal(x, y, message, rtol=1e-5):
    assert np.allclose(x, y, rtol=1e-5), message

def test_basex_basis_set():
    """
    Check that the basis.py returns the same result as the BASIS1.m script
    """
    size = 101
    M_ref, Mc_ref = basex_parse_matlab_basis_sets(os.path.join(DATA_DIR, 'dan_basis100{}_1.bst.gz'))

    M, Mc = basex_generate_basis_sets(size, nbf='auto', verbose=False)

    yield assert_allclose, Mc_ref, Mc, 1e-7, 1e-100
    yield assert_allclose, M_ref, M, 1e-7, 1e-100

def test_basex_basis_sets_cache():
    n = 121
    file_name = os.path.join(DATA_DIR, "basex_basis_{}_{}.npy".format(n, n//2))
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    basex_get_basis_sets_cached(n, basis_dir=DATA_DIR, verbose=False)
    # 2nd call load from file
    basex_get_basis_sets_cached(n, basis_dir=DATA_DIR, verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)


def test_basex_step_ratio():
    # This test checks that 

    n = 101
    r_max = 25

    step_options = dict(A0=10.0, r1=6.0, r2=14.0, ratio_valid_step=0.6)

    ref = StepAnalytical(n, r_max, symmetric=True, **step_options)


    # Calculate the inverse abel transform for the centered data
    inv_ab = BASEX(n=n, basis_dir=None, verbose=False, calc_speeds=False, dr=ref.dr)
    recon = inv_ab(ref.abel, center=n//2 , median_size=2,
                        gaussian_blur=0, post_median=0)


    ratio_mean, ratio_std, _ = absolute_ratio_benchmark(ref, recon)
    backend_name = type(inv_ab).__name__

    yield assert_allclose, ratio_mean, 1.0, 1e-2, 0, "{}: ratio == 1.0".format(backend_name)
    yield assert_allclose, ratio_std, 0.0,  1e-5, 4e-2,  "{}: std == 0.0".format(backend_name)

