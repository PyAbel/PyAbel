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

try:
    from abel.direct import fabel_direct, iabel_direct
    skip_tests = False
except CythonExtensionsNotBuilt:
    skip_tests = True



def test_direct_shape():
    if skip_tests:
        raise SkipTest
    n = 21
    x = np.ones((n, n))

    recon = fabel_direct(x)

    assert recon.shape == (n, n) 

    recon = iabel_direct(x)

    assert recon.shape == (n, n)


def test_direct_zeros():
    # just a sanity check
    if skip_tests:
        raise SkipTest
    n = 64
    x = np.zeros((n,n))
    assert (fabel_direct(x)==0).all()

    assert (iabel_direct(x)==0).all()


def test_inverse_direct_gaussian():
    """Check iabel_direct with a Gaussian"""
    if skip_tests:
        raise SkipTest
    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=False,  sigma=10)

    recon = iabel_direct(ref.abel, dr=ref.dr)

    ratio = absolute_ratio_benchmark(ref, recon, kind='inverse')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)


def test_forward_direct_gaussian():
    """Check fabel_direct with a Gaussian"""
    if skip_tests:
        raise SkipTest
    n = 51
    r_max = 25

    ref = GaussianAnalytical(n, r_max, symmetric=False,  sigma=10)

    recon = fabel_direct(ref.func, dr=ref.dr)

    ratio = absolute_ratio_benchmark(ref, recon, kind='direct')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)
