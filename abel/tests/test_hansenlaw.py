from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose

import abel

# Curve A, Table 2, Fig 3. Abel transform pair  Hansen&Law JOSA A2 510 (1985)
def f(r):
    return 1-2*r**2 if np.all(r) <= 0.5 else 2*(1-r)**2

def g(R):
    R2 = R**2
    alpha = np.sqrt(1-R**2)

    if np.all(R) <= 0.5:
        beta  = np.sqrt(0.25-R**2)
        return (2/3)*(2*alpha*(1+2*R2)-beta*(1+8*R2))-\
               4*R2*np.log((1+alpha)/(0.5+beta))
    else:
        return (4/3)*alpha*(1+2*R2)-4*R2*np.log((1+alpha)/R)


def test_hansenlaw_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction='inverse')

    assert recon.shape == (n, n) 


def test_hansenlaw_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.hansenlaw.hansenlaw_transform(x, direction="inverse")

    assert_allclose(recon, 0)


def test_hansenlaw_forward_tansform_gaussian():
    """Check hansenlaw forward tansform with a Gaussian function"""
    n = 1001
    r_max = 501   # more points better fit

    ref = abel.tools.analytical.GaussianAnalytical(n, 
          r_max, symmetric=False,  sigma=200)

    recon = abel.hansenlaw.hansenlaw_transform(ref.func, ref.dr,
                                               direction='forward')

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon, kind='direct')

    assert_allclose(ratio, 1.0, rtol=7e-2, atol=0)


def test_hansenlaw_inverse_transform_gaussian():
    """Check hansenlaw inverse transform with a Gaussian function"""
    n = 1001   # better with a larger number of points
    r_max = 501

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, 
          symmetric=False,  sigma=200)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    recon = abel.hansenlaw.hansenlaw_transform(tr, ref.dr, direction='inverse')
    recon1d = recon[n//2 + n%2]  # centre row

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)

    assert_allclose(ratio, 1.0, rtol=1e-1, atol=0)


def test_hansenlaw_forward_curveA():
    """ Check hansenlaw forward transform with 'curve A'
    """
    delta = 0.01  # sample size

    # split r-domain to suit function pair
    rl = np.arange(0, 0.5+delta/2, delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta, 1.0, delta) # 0.5 < r < 1.0
    r  = np.concatenate((rl,rr), axis=0) # whole r = [0,1)

    orig = np.concatenate((f(rl),f(rr)), axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)), axis=0)   # g(r)

    # forward Abel 
    Aproj = abel.hansenlaw.hansenlaw_transform(orig, delta,
                                               direction='forward')  
                                                       # == g(r)
    assert_allclose(proj, Aproj, rtol=0, atol=6.0e-2)


def test_hansenlaw_inverse_transform_curveA():
    """ Check hansenlaw inverse transform() 'curve A'
    """
    delta = 0.001 # sample size, smaller the better inversion

    # split r-domain to suit function pair
    rl = np.arange(0, 0.5+delta/2, delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta, 1.0, delta) # 0.5 < r < 1.0
    r  = np.concatenate((rl,rr), axis=0) # whole r = [0,1)

    orig = np.concatenate((f(rl),f(rr)), axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)), axis=0)   # g(r)

    # inverse Abel 
    recon = abel.hansenlaw.hansenlaw_transform(proj, r[1]-r[0],
                                               direction='inverse') 
                                                       # == f(r)
    assert_allclose(orig, recon, rtol=0, atol=0.01)


def test_hansenlaw_forward_dribinski_image():
    """ Check hansenlaw forward/inverse transform
        using BASEX sample image, comparing speed distributions
    """

    # BASEX sample image
    IM = abel.tools.analytical.sample_image(n=1001, name="dribinski")

    # core transform(s) use top-right quadrant, Q0
    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(IM)

    # forward Abel transform
    fQ0 = abel.hansenlaw.hansenlaw_transform(Q0, direction='forward')

    # inverse Abel transform
    ifQ0 = abel.hansenlaw.hansenlaw_transform(fQ0, direction='inverse')
    
    # speed distribution
    orig_speed, orig_radial = abel.tools.vmi.angular_integration(Q0, 
                              origin=(0,0), Jacobian=True)

    speed, radial_coords = abel.tools.vmi.angular_integration(ifQ0,
                           origin=(0,0), Jacobian=True)

    orig_speed /= orig_speed[50:125].max()
    speed /= speed[50:125].max()
    

    assert np.allclose(orig_speed[50:125], speed[50:125], rtol=0.5, atol=0)

if __name__ == "__main__":
    test_hansenlaw_shape()
    test_hansenlaw_zeros()
    test_hansenlaw_forward_tansform_gaussian()
    test_hansenlaw_inverse_transform_gaussian()
    test_hansenlaw_forward_curveA()
    test_hansenlaw_inverse_transform_curveA()
    test_hansenlaw_forward_dribinski_image()
