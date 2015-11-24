#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright CNRS 2012
# Roman Yurchak (LULI)
# This software is governed by the CeCILL-B license under French law and
# abiding by the rules of distribution of free software.
import time

import numpy as np
from abel.direct import abel_transform, iabel_transform
from abel.math import gradient
import scipy.ndimage as nd
from numpy.testing import assert_allclose


def test_direct_zeros():
    # just a sanity check
    n = 64
    x = np.zeros((n,n))
    assert (abel_transform(x)==0).all()


def test_direct_gaussian():
    n = 500
    r = np.linspace(0, 5., n)
    dr = np.diff(r)[0]
    rc = 0.5*(r[1:]+r[:-1])
    fr = np.exp(-rc**2)
    Fn = abel_transform(fr, dr=dr)
    Fn_a = np.pi**0.5*np.exp(-rc**2)
    yield assert_allclose,  Fn, Fn_a, 1e-2, 1e-3



def test_direct_step():
    n = 800
    r = np.linspace(0, 20, n)
    dr = np.diff(r)[0]
    rc = 0.5*(r[1:]+r[:-1])
    fr = np.exp(-rc**2)
    #fr += 1e-1*np.random.rand(n)
#    plt.plot(rc,fr,'b', label='Original signal')
    F = abel_transform(fr, dr=dr)
    F_a = (np.pi)**0.5*fr.copy()

    F_i = iabel_transform(F, dr=dr, derivative=np.gradient)

    #yield assert_allclose, fr, F_i, 5e-3, 1e-6, 'Test that direct>inverse Abel equals the original data'
    #yield assert_allclose, F_a, F, 5e-3, 1e-6, 'Test direct Abel transforms failed!'

    yield assert_allclose, fr, F_i, 5e-2, 1e-6, 'Test that direct>inverse Abel equals the original data'
    yield assert_allclose, F_a, F, 5e-3, 1e-6, 'Test direct Abel transforms failed!'
