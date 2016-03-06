from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel

def test_symmetry_get_put_quadrants(image_shape=(5,5), verbose=False):
    
    n, m = image_shape
    z = np.ones((n, m))*(-1)     # odd-image axes tagged with '-1'
    c2 = m//2 
    r2 = n//2 
    # tag each quadrant with its number
    z[:r2, -c2:] = 0
    z[:r2, :c2] = 1
    z[-r2:, :c2] = 2
    z[-r2:, -c2:] = 3 

    if verbose:
        print("test image of shape ", z.shape)
        if n % 2 or m % 2:
            print(" odd-size axes tagged with '-1'")
        print(z)

    q = abel.tools.symmetry.get_image_quadrants(z, reorient=True)

    if verbose:
        for i, qi in enumerate(q):
            print("\nreoriented quadrant Q{:d}".format(i))
            print(qi)

    r = abel.tools.symmetry.put_image_quadrants(q, original_image_shape=z.shape)

    if verbose:
        print("\nreassembled image, shape = ", r.shape)
        print(r)

    assert not (z-r).any()

"""
    q = abel.tools.symmetry.get_image_quadrants(z, reorient=True, 
            use_quadrants=(True, False, False, False), symmetry_axis=(0, 1)) 

    if verbose:
        for i, qi in enumerate(q):
            print("\nreoriented quadrant Q{:d}".format(i))
            print(qi)
"""


if __name__ == "__main__":
  test_symmetry_get_put_quadrants(image_shape=(5, 5), verbose=True)
