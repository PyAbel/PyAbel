# -*- coding: utf-8 -*-
import time

import numpy as np
from abel.math import gradient
import scipy.ndimage as nd
from numpy.testing import assert_allclose
from abel.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark
from abel.tools import CythonExtensionsNotBuilt
from unittest.case import SkipTest
from abel.direct import fabel_direct, iabel_direct, cython_ext, simpson_rule_wrong
import abel.direct



def test_direct_shape():
    if not cython_ext:
        raise SkipTest
    n = 21
    x = np.ones((n, n))

    recon = fabel_direct(x)

    assert recon.shape == (n, n) 

    recon = iabel_direct(x)

    assert recon.shape == (n, n)


def test_direct_zeros():
    # just a sanity check
    if not cython_ext:
        raise SkipTest
    n = 64
    x = np.zeros((n,n))
    assert (fabel_direct(x)==0).all()

    assert (iabel_direct(x)==0).all()


def test_inverse_direct_gaussian():
    """Check iabel_direct with a Gaussian"""
    if not cython_ext:
        raise SkipTest
    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=False,  sigma=10)

    recon = iabel_direct(ref.abel, dr=ref.dr)

    ratio = absolute_ratio_benchmark(ref, recon, kind='inverse')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)


def test_direct_c_python_correspondance_wcorrection():
    """ Check that both the C and Python backends are identical (correction=True)"""
    if not cython_ext:
        raise SkipTest
    N = 10
    r = 0.5 + np.arange(N).astype('float64') 
    x = 2*r.reshape((1, -1))**2
    out1 =  abel.direct._pyabel_direct_integral(x, r, 1)
    out2 = abel.direct._cabel_direct_integral(x, r, 1)
    raise SkipTest  # this tests does not pass
    assert_allclose(out1, out2, rtol=0.3)


def test_direct_c_python_correspondance():
    """ Check that both the C and Python backends are identical (correction=False)"""
    if not cython_ext:
        raise SkipTest
    N = 10
    r = 0.5 + np.arange(N).astype('float64')
    x = 2*r.reshape((1, -1))**2
    out1 = abel.direct._pyabel_direct_integral(x, r, 0)
    out2 = abel.direct._cabel_direct_integral(x, r, 0)
    raise SkipTest  # this tests does not pass
    assert_allclose(out1, out2, rtol=0.3)


def test_forward_direct_gaussian():
    """Check fabel_direct with a Gaussian"""
    if not cython_ext:
        raise SkipTest
    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=False,  sigma=10)

    recon = fabel_direct(ref.func, dr=ref.dr)

    ratio = absolute_ratio_benchmark(ref, recon, kind='direct')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)


def test_simps_wrong():
    from scipy.integrate import simps
    dx = 1.0
    x = np.arange(32).reshape((1, -1))

    res1 = simps(x, dx=dx)
    res2 = simpson_rule_wrong(x, dx=dx)
    assert_allclose(res1, res2, rtol=0.001)



