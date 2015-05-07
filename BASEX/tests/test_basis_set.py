#!/usr/bin/python
import os.path

import numpy as np
from numpy.testing import assert_allclose


from BASEX.core import get_left_right_matrices
from BASEX.io import parse_matlab
from BASEX.basis import generate_basis_sets

DATA_DIR = os.path.join(os.path.split(__file__)[0], '../data/')

def setup():
    pass



#def test_consistency_included_dataset():
#    # just a sanity check
#    path = os.path.join(DATA_DIR, 'basex_basis_1000_500_orig.npy')
#
#    left, right, M, Mc = np.load(path)
#
#    left_new, right_new = get_left_right_matrices(M, Mc)
#
#    # checking that get_left_right is consistent with the shipped data
#    yield assert_allclose, left, left_new
#    yield assert_allclose, right, right_new

    #Ni, Nj = M.shape
    ##rawdata = np.random.randn(Ni, Ni).view(np.matrix)

    #Ci = (left*rawdata)*right
    #P = (Mc*Ci)*M.T
    #print(P)
    #print(rawdata)
    #yield assert_allclose, rawdata, P

def test_basis_original():
    """
    Comparing the basis generated with "BASIS1.m"  with the basis set originally included with this package
    """
    M, Mc = parse_matlab(os.path.join(DATA_DIR, 'ascii', 'dan_basis1000{}_1.bst.gz'))
    M_orig, Mc_orig = parse_matlab(os.path.join(DATA_DIR, 'ascii', 'original_basis1000{}_1.txt.gz'))

    yield assert_allclose, Mc_orig, Mc, 1e-7, 1e-50
    yield assert_allclose, M_orig, M, 1e-7, 1e-50


def test_generation_basis():
    """
    Check the that the basis.py returns the same result as the BASIS1.m script
    """
    size = 1000
    M_ref, Mc_ref = parse_matlab(os.path.join(DATA_DIR, 'ascii', 'dan_basis1000{}_1.bst.gz'))

    M, Mc = generate_basis_sets(size+1, size//2)

    yield assert_allclose, Mc_ref, Mc, 1e-7, 1e-100
    yield assert_allclose, M_ref, M, 1e-7, 1e-100


#def test_generation_basis100():
#    left, right, M_ref, Mc_ref = np.load(os.path.join(DATA_DIR, 'basex_basis_1000x1000.npy'))
#
#    M, Mc = generate_basis(101, 50, verbose=True)
#
#
#    yield assert_allclose, Mc_ref.view(np.ndarray), Mc.view(np.ndarray)
#    yield assert_allclose, M_ref.view(np.ndarray), M.view(np.ndarray)



