from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.basex import BASEX
from abel.io import parse_matlab
from abel.basex import generate_basis_sets
from abel.benchmark import SymStepBenchmark


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_basex_basis_set():
    """
    Check that the basis.py returns the same result as the BASIS1.m script
    """
    size = 100
    M_ref, Mc_ref = parse_matlab(os.path.join(DATA_DIR, 'dan_basis100{}_1.bst.gz'))

    M, Mc = generate_basis_sets(size+1, size//2, verbose=False)

    yield assert_allclose, Mc_ref, Mc, 1e-7, 1e-100
    yield assert_allclose, M_ref, M, 1e-7, 1e-100

def test_basex_step():
    """
    Check that BASEX implementation passes the SymStepBenchmark (up to a scaling factor)
    """

    n = 101
    r_max = 25
    A0 = 10.0
    r1 = 6.0
    r2 = 14.0

    sbench = SymStepBenchmark(n, r_max, r1, r2, A0)

    st = sbench.step

    # Calculate the inverse abel transform for the centered data
    inv_ab = BASEX(n=n, nbf=n//2, basis_dir=None, verbose=False, calc_speeds=False)
    center = n//2
    recon = inv_ab(st.abel, center , median_size=2,
                        gaussian_blur=0, post_median=0)


    err_mean, err_std, _ = sbench.run(recon, 0.5)

    snr = err_mean/err_std # calculate a signal to noise ratio (SNR)
    assert snr > 100  # a randomly large value
