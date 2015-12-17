from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

from abel.hansenlaw import fabel_hansenlaw_transform, iabel_hansenlaw,\
                           iabel_hansenlaw_transform, fabel_hansenlaw
from abel.analytical import GaussianAnalytical
from abel.benchmark import absolute_ratio_benchmark

# Curve A, Table 2, Fig 3. Abel transform pair  Hansen&Law JOSA A2 510 (1985)
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
    """ Check hansenlaw_transform() curve A
    """
    delta = 0.01 # sample size

    # split r-domain to suit function pair
    # there must be a more Python way to achieve this
    rl = np.arange(0,0.5+delta/2,delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta,1.0,delta) # 0.5 < r < 1.0
    r = np.concatenate((rl,rr),axis=0)  # whole r = [0,1)

    orig = np.concatenate((f(rl),f(rr)),axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)),axis=0)   # g(r)
    orig = orig[::-1] # flip them  
    proj = proj[::-1]

    recon = iabel_hansenlaw_transform(proj)[0]  # inverse Abel 
                                                # == f(r)???
    orig = orig[::-1]  # flip back
    recon = recon[::-1]/delta  # flip + scaling for sample size not 1

    mask = r < 0.9  # check deviation away from small values
    assert_allclose(orig[mask], recon[mask], rtol=0.1, atol=0)

def test_fabel_hansenlaw():
    """ Check fabel_hansenlaw_transform() curve A
    """
    delta = 0.01 # sample size

    # split r-domain to suit function pair
    # there must be a more Python way to achieve this
    rl = np.arange(0,0.5+delta/2,delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta,1.0,delta) # 0.5 < r < 1.0
    r = np.concatenate((rl,rr),axis=0)  # whole r = [0,1)

    orig = np.concatenate((f(rl),f(rr)),axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)),axis=0)   # g(r)
    orig = orig[::-1] # flip them  
    proj = proj[::-1]

    Aproj = fabel_hansenlaw_transform(orig)[0]  # forward Abel 
                                                # == g(r)???
    orig = orig[::-1]  # flip back
    Aproj = Aproj*delta*np.pi  # no flip + scaling for sample size not 1

    mask = np.logical_and(r > 0.1,r<0.9)  
    assert_allclose(proj[mask],Aproj[mask], rtol=0.1, atol=0)


def test_forward_direct_gaussian():
    """Check fabel_direct with a Gaussian"""
    n = 501
    r_max = 251

    ref = GaussianAnalytical(n, r_max, symmetric=False,  sigma=100)

    recon = fabel_hansenlaw_transform(ref.func)[0]

    ratio = absolute_ratio_benchmark(ref, recon, kind='direct')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)

