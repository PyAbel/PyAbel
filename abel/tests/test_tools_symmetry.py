#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import numpy as np
from numpy.testing import assert_allclose

import abel

def test_symmetry_get_put_quadrants():
    
    z = np.zeros((5, 5))
    z[1, 1] = 1
    z[1, -2] = 1
    z[-2, 1] = 1
    z[-2, -2]= 1

    q = abel.tools.symmetry.get_image_quadrants(z, reorient=True)

    r = abel.tools.symmetry.put_image_quadrants(q)


    assert not (z-r).any()


if __name__ == "__main__":
  test_symmetry_get_put_quadrants()
