from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel

def test_symmetry_get_put_quadrants(n=5, verbose=False):
    
    z = np.ones((n, n))*(-1)     # axes tagged with '-1'
    c2 = n//2 
    r2 = n//2 
    # tag each quadrant with its number
    z[:r2, -c2:] = 0
    z[:r2, :c2] = 1
    z[-r2:, :c2] = 2
    z[-r2:, -c2:] = 3 

    if verbose:
        print("test image")
        print(z)

    q = abel.tools.symmetry.get_image_quadrants(z, reorient=True)

    if verbose:
        for i, qi in enumerate(q):
            print("\nreoriented quadrant Q{:d}".format(i))
            print(qi)

    r = abel.tools.symmetry.put_image_quadrants(q, original_image_shape=(n,n))

    if verbose:
        print("\nreassembled image")
        print(r)


    assert not (z-r).any()


if __name__ == "__main__":
  test_symmetry_get_put_quadrants(n=5, verbose=True)
