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

def test_onion_dasch_basis_sets_cache():
    n = 121
    file_name = os.path.join(DATA_DIR, "onion_dasch_basis_{}_{}.npy".format(n, n))
    if os.path.exists(file_name):
        os.remove(file_name)
    # 1st call generate and save
    abel.tools.basis.get_bs_cached("onion_dasch", n, basis_dir=DATA_DIR,
                                verbose=False)
    # 2nd call load from file
    abel.tools.basis.get_bs_cached("onion_dasch", n, basis_dir=DATA_DIR,
                                verbose=False)
    if os.path.exists(file_name):
        os.remove(file_name)

def test_onion_dasch_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.onion_dasch.onion_dasch_transform(x, direction='inverse')

    assert recon.shape == (n, n) 


def test_onion_dasch_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.onion_dasch.onion_dasch_transform(x, direction="inverse")

    assert_allclose(recon, 0)

def test_onion_dasch_inverse_transform_gaussian():
    """Check onion_dasch inverse transform with a Gaussian function"""
    n = 501   
    r_max = 251

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, 
          symmetric=False,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    recon = abel.onion_dasch.onion_dasch_transform(tr, basis_dir=None, dr=1)
    recon1d = recon[n//2 + n%2]  # centre row

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)/2

    assert_allclose(ratio, 1.0, rtol=1, atol=0)

def test_onion_dasch_1d_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    rows, cols = n, n
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2

    sigma = 20*n/100

    # 1D Gaussian -----------
    r = np.linspace(0, c2-1, c2)

    orig = gauss(r, 0, sigma)
    orig_copy = orig.copy()

    recon = abel.onion_dasch.onion_dasch_transform(orig)

    ratio_1d = np.sqrt(np.pi)*sigma

    assert_allclose(orig_copy[20:], recon[20:]*ratio_1d, rtol=0.0, atol=0.5)

def test_onion_dasch_2d_gaussian(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    image_shape=(n, n)
    rows, cols = image_shape
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2
    sigma = 20*n/100

    x = np.linspace(-c2, c2, cols)
    y = np.linspace(-r2, r2, rows)

    X, Y = np.meshgrid(x, y)

    IM = gauss(X, 0, sigma) # cylindrical Gaussian located at pixel R=0
    Q0 = IM[:r2, c2:] # quadrant, top-right
    Q0_copy = Q0.copy()

    # onion_dasch inverse Abel transform
    AQ0 = abel.onion_dasch.onion_dasch_transform(Q0)
    profQ0 = Q0_copy[-10:,:].sum(axis=0)
    profAQ0 = AQ0[-10:,:].sum(axis=0)

    ratio_2d = np.sqrt(np.pi)*sigma

    assert_allclose(Q0_copy, AQ0*ratio_2d, rtol=0.0, atol=0.3)

if __name__ == "__main__":
    test_onion_dasch_shape()
    test_onion_dasch_zeros()
    test_onion_dasch_1d_gaussian()
    test_onion_dasch_inverse_transform_gaussian()
    test_onion_dasch_2d_gaussian()
