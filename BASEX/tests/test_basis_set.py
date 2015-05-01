#!/usr/bin/python
import os.path

import numpy as np
from numpy.testing import assert_allclose


from BASEX.core import get_left_right
from BASEX.io import parse_matlab
from BASEX.basis import generate_basis

DATA_DIR = os.path.join(os.path.split(__file__)[0], '../data/')

def setup():
    pass



def test_consistency_included_dataset():
    # just a sanity check
    path = os.path.join(DATA_DIR, 'basex_basis_1000_orig.npy')

    left, right, M, Mc = np.load(path)

    left_new, right_new = get_left_right(M, Mc)

    # checking that get_left_right is consistent with the shipped data
    yield assert_allclose, left, left_new
    yield assert_allclose, right, right_new

    #Ni, Nj = M.shape
    ##rawdata = np.random.randn(Ni, Ni).view(np.matrix)

    #Ci = (left*rawdata)*right
    #P = (Mc*Ci)*M.T
    #print(P)
    #print(rawdata)
    #yield assert_allclose, rawdata, P

def test_generation_basis40():
    base_dir = os.path.join(DATA_DIR, 'ascii')
    for size in [40, 100]:
        M_ref, Mc_ref = parse_matlab('basis{}'.format(size), base_dir, gzip=False)

        M, Mc = generate_basis(size+1, size//2)

        yield assert_allclose, Mc_ref.view(np.ndarray), Mc
        yield assert_allclose, M_ref.view(np.ndarray), M


#def test_generation_basis100():
#    left, right, M_ref, Mc_ref = np.load(os.path.join(DATA_DIR, 'basex_basis_1000x1000.npy'))
#
#    M, Mc = generate_basis(101, 50, verbose=True)
#
#
#    yield assert_allclose, Mc_ref.view(np.ndarray), Mc.view(np.ndarray)
#    yield assert_allclose, M_ref.view(np.ndarray), M.view(np.ndarray)



