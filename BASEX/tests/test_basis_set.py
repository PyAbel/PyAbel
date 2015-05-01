#!/usr/bin/python
import os.path

import numpy as np
from numpy.testing import assert_allclose


from BASEX.core import get_left_right

DATA_DIR = os.path.join(os.path.split(__file__)[0], '../data/')

def setup():
    pass



def test_consistency_included_dataset():
    # just a sanity check
    path = os.path.join(DATA_DIR, 'basex_basis_1000x1000.npy')

    left, right, M, Mc = np.load(path)

    left_new, right_new = get_left_right(M, Mc)

    # checking that get_left_right is consistent with the shipped data
    yield assert_allclose, left, left_new
    yield assert_allclose, right, right_new

    Ni, Nj = M.shape
    rawdata = np.random.randn(Ni, Ni).view(np.matrix)
    print(left.shape)
    print(right.shape)
    print(M.shape)
    print(Mc.shape)

    Ci = (left*rawdata)*right
    P = (Mc*Ci)*M.T
    print(P)
    print(rawdata)
    yield assert_allclose, rawdata, P




