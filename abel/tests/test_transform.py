# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.testing import assert_allclose
import abel

# The role of abel.transform() is to:
# 0. (optional) center image and make odd-width
# 1. split an image into quadrants,
# 2. transform each quadrant
# 3. reassemble image
# 4. (optional) evaluate speed profile
# The specific functions will have been tested separately
# here we check that the functions combine correctly

def test_transform_shape():
    Zeven = np.ones((4,4))
    Zodd = np.ones((4,5))

    try:
        # even width image should be rejected
        iZ = abel.transform(Zeven) 
        # fail if get here
        assert iZ['transform'].shape[1] % 2 == 1
    except:
        pass

    iZ = abel.transform(Zodd) 
    # check odd width
    assert iZ['transform'].shape[1] % 2 == 1

    iZ = abel.transform(Zeven, center="com")
    # check odd width
    assert iZ['transform'].shape[1] % 2 == 1


def test_transform_angular_integration(n=101):
    gauss = lambda r, r0, sigma: np.exp(-(r-r0)**2/sigma**2)

    image_shape=(n, n)
    rows, cols = image_shape
    r2 = rows//2 + rows % 2
    c2 = cols//2 + cols % 2
    x = np.linspace(-c2, c2, cols)
    y = np.linspace(-r2, r2, rows)

    X, Y = np.meshgrid(x, y)
    R, THETA = abel.tools.polar.cart2polar(X, Y)

    IM = gauss(R, c2, 2) # Gaussian donut located at R=c2

    # forward Abel transform
    fIM = abel.transform(IM, method="hansenlaw",
		    direction="forward")['transform']

    # inverse Abel transform, and radial intensity distribution
    results = abel.transform(fIM, method="hansenlaw", direction="inverse",
                             angular_integration=True)

    assert 'transform' in results.keys()
    assert 'angular_integration' in results.keys()

    radial = results['angular_integration']
    assert np.shape(radial) == (2, int(np.sqrt(c2**2+r2**2))) 

    max_position = radial[1].argmax()
    assert np.allclose(max_position, c2, rtol=0, atol=2) 
        

if __name__ == "__main__":
    test_transform_shape()
    test_transform_angular_integration()
