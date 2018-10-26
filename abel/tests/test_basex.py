from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel
from abel.tools.analytical import GaussianAnalytical


DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')

def test_basex_basis_sets_cache():
    # n = 61  (121 full width)
    # sigma = 1  (nbf = 61)
    n = 61
    sigma = 1.0
    file_name = os.path.join(DATA_DIR,
                             "basex_basis_{}_{}.npy".format(n, sigma))
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


def basex_gaussian(sigma, reg, cor, tol):
    """Check a gaussian solution for BASEX"""
    n = 100
    r_max = n - 1

    ref = GaussianAnalytical(n, r_max, symmetric=False, sigma=30)
    tr = np.tile(ref.abel[None, :], (n, 1))  # make a 2D array from 1D

    correction = cor if isinstance(cor, bool) else False

    Ai = abel.basex.get_bs_basex_cached(n, sigma=sigma, reg=reg,
                                        correction=correction,
                                        basis_dir=None, verbose=False)

    recon = abel.basex.basex_core_transform(tr, Ai)
    recon = recon[n // 2 + n % 2]

    ref = ref.func
    if not isinstance(cor, bool):
        # old-style intensity correction
        recon /= cor
        # skip artifact from k = 0 near r = 0
        # see https://github.com/PyAbel/PyAbel/issues/230
        cut = int(2 * sigma)
        recon = recon[cut:]
        ref = ref[cut:]

    assert_allclose(recon, ref, atol=tol)


def test_basex_gaussian():
    """Check a gaussian solution for BASEX:
       default parameters"""
    # (intensity correction using "magic number",
    #  see https://github.com/PyAbel/PyAbel/issues/230)
    basex_gaussian(sigma=1, reg=0, cor=1.015, tol=3e-3)


def test_basex_gaussian_corrected():
    """Check a gaussian solution for BASEX:
       default parameters, corrected"""
    basex_gaussian(sigma=1, reg=0, cor=True, tol=7e-4)


def test_basex_gaussian_sigma_3():
    """Check a gaussian solution for BASEX:
       large sigma (oscillating)"""
    basex_gaussian(sigma=3, reg=0, cor=1, tol=3e-2)


def test_basex_gaussian_sigma_3_corrected():
    """Check a gaussian solution for BASEX:
       large sigma (oscillating)"""
    basex_gaussian(sigma=3, reg=0, cor=True, tol=2e-3)


def test_basex_gaussian_sigma_07_reg_10_corrected():
    """Check a gaussian solution for BASEX:
       small sigma, regularized, corrected"""
    basex_gaussian(sigma=0.7, reg=10, cor=True, tol=6e-3)


if __name__ == '__main__':
    test_basex_basis_sets_cache()
    test_basex_shape()
    test_basex_zeros()
    test_basex_gaussian()
    test_basex_gaussian_corrected()
    test_basex_gaussian_sigma_3()
    test_basex_gaussian_sigma_3_corrected()
    test_basex_gaussian_sigma_07_reg_10_corrected()
