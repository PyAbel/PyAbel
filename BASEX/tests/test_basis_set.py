#!/usr/bin/python
import os.path

import numpy as np
from numpy.testing import assert_allclose


from BASEX.core import get_left_right, parse_matlab
from BASEX.basis import generate_basis

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

    #Ni, Nj = M.shape
    ##rawdata = np.random.randn(Ni, Ni).view(np.matrix)
    #print(left.shape)
    #print(right.shape)
    #print(M.shape)
    #print(Mc.shape)

    #Ci = (left*rawdata)*right
    #P = (Mc*Ci)*M.T
    #print(P)
    #print(rawdata)
    #yield assert_allclose, rawdata, P

def test_basis_generation():
    base_dir = os.path.join(DATA_DIR, 'ascii')
    M_ref, Mc_ref = parse_matlab('basis40', base_dir)

    M, Mc = generate_basis(41, 20)

    yield assert_allclose, Mc_ref.view(np.ndarray), Mc
    yield assert_allclose, M_ref.view(np.ndarray), M





