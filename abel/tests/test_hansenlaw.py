from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.hansenlaw import  iabel_hansenlaw, iabel_hansenlaw_transform
from abel.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark


def test_hansenlaw_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = iabel_hansenlaw(x, calc_speeds=False, verbose=False)

    assert recon.shape == (n, n) 


def test_hansenlaw_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = iabel_hansenlaw(x, calc_speeds=False, verbose=False)

    assert_allclose(recon, 0)


def test_hansenlaw_gaussian():
    """Check a gaussian solution for HansenLaw"""
    n = 1001   # better with a larger number of points
    r_max = 501

    ref = GaussianAnalytical(n, r_max, symmetric=True,  sigma=200)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D


    recon = iabel_hansenlaw(tr)
    recon1d = recon[n//2 + n%2]  # centre row

    ratio = absolute_ratio_benchmark(ref, recon1d)

    assert_allclose(ratio,  1.0, rtol=0.1, atol=0)

def test_hansenlaw_curveA():
    """ Check hansenlaw_transform()
        verify curve A, Table 2, Fig. 3, JOSA A2 510 (1985).
    """
    # Abel transform pair
    def f (r):  
        return 1-2*r**2 if np.all(r) <= 0.5 else 2*(1-r)**2

    def g (R):  
        alpha = np.sqrt(1-R**2)
        R2 = R**2

        if np.all(R) <= 0.5:
            beta  = np.sqrt(0.25-R**2)
            return (2/3)*(2*alpha*(1+2*R2)-beta*(1+8*R2))-\
                   4*R2*np.log((1+alpha)/(0.5+beta))
        else:
            return (4/3)*alpha*(1+2*R2)-4*R2*np.log((1+alpha)/R)

    delta = 0.01
    # split r-range to suit function pair
    # there must be a more Python way to achieve this
    rl = np.arange(0,0.5+delta/2,delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta,1.0,delta) # 0.5 < r < 1.0
    r = np.concatenate((rl,rr),axis=0)  # r = [0,1)

    orig = np.concatenate((f(rl),f(rr)),axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)),axis=0)   # g(r)
    orig = orig[::-1] # flip them  
    proj = proj[::-1]

    recon = iabel_hansenlaw_transform(proj)[0]  # inverse Abel 
                                                # == g(r)???
    orig = orig[::-1]  # flip back
    recon = recon[::-1]/delta

    mask = r > 0.1  # check deviation away from "image centre"
    diff = np.abs(recon[mask]-orig[mask])
    assert_allclose(diff, 0.0, atol=0.011)



