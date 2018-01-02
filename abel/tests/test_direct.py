# -*- coding: utf-8 -*-
import time

import numpy as np
from unittest.case import SkipTest
from numpy.testing import assert_allclose
import scipy.ndimage as nd
import abel

def test_direct_shape():
    if not abel.direct.cython_ext:
        raise SkipTest
    n = 21
    x = np.ones((n, n))

    recon = abel.direct.direct_transform(x, direction='forward')
    assert recon.shape == (n, n) 

    recon = abel.direct.direct_transform(x, direction='inverse')
    assert recon.shape == (n, n)


def test_direct_zeros():
    # just a sanity check
    if not abel.direct.cython_ext:
        raise SkipTest
    n = 64
    x = np.zeros((n,n))
    assert (abel.direct.direct_transform(x, direction='forward')==0).all()
    assert (abel.direct.direct_transform(x, direction='inverse')==0).all()


def test_direct_gaussian():
    """Check abel.direct.direct_transform() with a Gaussian"""
    if not abel.direct.cython_ext:
        raise SkipTest
    n = 501
    r_max = 100

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, symmetric=False,
                                                   sigma=20)

    # forward: 
    recon = abel.direct.direct_transform(ref.func, dr=ref.dr, direction='forward')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='forward')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)
    
    # inverse:
    recon = abel.direct.direct_transform(ref.abel, dr=ref.dr, direction='inverse')
    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='inverse')
    assert_allclose(ratio, 1.0, rtol=2e-2, atol=2e-2)
    

def test_direct_c_python_correspondance_wcorrection():
    """ Check that both the C and Python backends are identical (correction=True)"""
    if not abel.direct.cython_ext:
        raise SkipTest
    N = 10
    r = 0.5 + np.arange(N).astype('float64') 
    x = 2*r.reshape((1, -1))**2
    out1 =  abel.direct._pyabel_direct_integral(x, r, 1)
    out2 =  abel.direct._cabel_direct_integral( x, r, 1)
    assert_allclose(out1, out2, rtol=1e-9, atol=1e-9)


def test_direct_c_python_correspondance():
    """ Check that both the C and Python backends are identical (correction=False)"""
    if not abel.direct.cython_ext:
        raise SkipTest
    N = 10
    r = 0.5 + np.arange(N).astype('float64')
    x = 2*r.reshape((1, -1))**2
    
    out1 = abel.direct._pyabel_direct_integral(x, r, 0)
    out2 = abel.direct._cabel_direct_integral( x, r, 0)
    assert_allclose(out1, out2, rtol=1e-9, atol=1e-9)


if __name__ == "__main__":
    test_direct_shape()
    test_direct_zeros()
    test_direct_gaussian()
    test_direct_c_python_correspondance_wcorrection()
    test_direct_c_python_correspondance()
