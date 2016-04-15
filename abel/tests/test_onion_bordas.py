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


def test_onion_bordas_shape():
    n = 21
    x = np.ones((n, n), dtype='float32')

    recon = abel.onion_bordas.onion_bordas_transform(x, direction='inverse')

    assert recon.shape == (n, n) 


def test_onion_bordas_zeros():
    n = 21
    x = np.zeros((n, n), dtype='float32')

    recon = abel.onion_bordas.onion_bordas_transform(x, direction="inverse")

    assert_allclose(recon, 0)


def test_onion_bordas_inverse_transform_gaussian():
    """Check onion_bordas inverse transform with a Gaussian function"""
    n = 501   
    r_max = 251

    ref = abel.tools.analytical.GaussianAnalytical(n, r_max, 
          symmetric=False,  sigma=10)
    tr = np.tile(ref.abel[None, :], (n, 1)) # make a 2D array from 1D

    recon = abel.onion_bordas.onion_bordas_transform(tr, ref.dr, direction='inverse')
    recon1d = recon[n//2 + n%2]  # centre row

    ratio = abel.benchmark.absolute_ratio_benchmark(ref, recon1d)/2

    assert_allclose(ratio, 1.0, rtol=1, atol=0)

def test_onion_bordas_inverse_transform_curveA():
    """ Check onion_bordas inverse transform() 'curve A'
    """
    delta = 0.01 

    # split r-domain to suit function pair
    rl = np.arange(0, 0.5+delta/2, delta) # 0 <= r <= 0.5
    rr = np.arange(0.5+delta, 1.0, delta) # 0.5 < r < 1.0
    r  = np.concatenate((rl,rr), axis=0) # whole r = [0,1)

    orig = np.concatenate((f(rl),f(rr)), axis=0)   # f(r)
    proj = np.concatenate((g(rl),g(rr)), axis=0)   # g(r)

    # inverse Abel 
    recon = abel.onion_bordas.onion_bordas_transform(proj, r[1]-r[0],
                                               direction='inverse') 
                                                       # == f(r)
    ratio = orig[20]/recon[20]

    assert_allclose(orig, recon*ratio, rtol=0, atol=0.1)

def test_onion_bordas_1d_gaussian(n=100):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    n = 100
    rows, cols = n, n
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2

    sigma = 20*n/100

    # 1D Gaussian -----------
    r = np.linspace(0, c2-1, c2)

    orig = gauss(r, 0, sigma)
    orig_copy = orig.copy()

    recon = abel.onion_bordas.onion_bordas_transform(orig, shift_grid=False)

    ratio_1d = np.sqrt(np.pi)*sigma

    assert_allclose(orig_copy[20:], recon[20:]*ratio_1d, rtol=0.0, atol=0.5)

def test_onion_bordas_2d_gaussian(n=100):
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

    # onion_bordas inverse Abel transform
    AQ0 = abel.onion_bordas.onion_bordas_transform(Q0, shift_grid=False)
    profQ0 = Q0_copy[-10:,:].sum(axis=0)
    profAQ0 = AQ0[-10:,:].sum(axis=0)

    ratio_2d = np.sqrt(np.pi)*sigma

    assert_allclose(Q0_copy, AQ0*ratio_2d, rtol=0.0, atol=0.3)

if __name__ == "__main__":
    test_onion_bordas_shape()
    test_onion_bordas_zeros()
    test_onion_bordas_1d_gaussian()
    test_onion_bordas_inverse_transform_gaussian()
    test_onion_bordas_inverse_transform_curveA()
    test_onion_bordas_2d_gaussian()
