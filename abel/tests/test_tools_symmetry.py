from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel

def test_symmetry_get_put_quadrants(verbose=False):
    
    z = np.zeros((5, 5))
    z[0,0] = 1
    z[-1,0] = 2
    z[-1,-1] = 3
    z[1, 1] = 1
    z[1, -2] = 1
    z[-2, 1] = 1
    z[-2, -2]= 1

    if verbose:
        print("test image")
        print(z)

    q = abel.tools.symmetry.get_image_quadrants(z, reorient=True)

    if verbose:
        for i, qi in enumerate(q):
            print("\nreoriented quadrant Q{:d}".format(i))
            print(qi)

    r = abel.tools.symmetry.put_image_quadrants(q, odd_size=True)

    if verbose:
        print("\nreassembled image")
        print(r)


    assert not (z-r).any()


if __name__ == "__main__":
  test_symmetry_get_put_quadrants(verbose=True)
