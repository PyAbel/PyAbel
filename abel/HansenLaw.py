#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from time import time
from math import exp, log, pow, pi
from abel.tools import calculate_speeds

######################################################################
# HasenLaw - an alternative inverse Abel transformation algorithm 
#            to basex
#
# Stephen Gibson - Australian National University, Australia
# Jason Gascooke - Flinders University, Australia
# 
# This algorithm is adapted by Jason Gascooke from the article
#   E. W. Hansen and P-L. Law
#  "Recursive methods for computing the Abel transform and its inverse"
#  J. Opt. Soc. Am A2, 510-520 (1985)
#
# J. R. Gascooke PhD Thesis:
#  "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals 
#   Molecule Dissociation", Flinders University, 2000.
#
# Implemented in Python, with image quadrant co-adding, by Steve Gibson
#
# Note: the inversion is carried out one image row at a time
#
########################################################################

def HansenLaw_transform (ImgRow):
    """ Hansen and Law J. Opt. Soc. Am A2, 510-520 (1985)
        much simpler than BASEX!
    """
    h   = [0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3]
    lam = [0.0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1]
    Gamma = lambda Nm, lam: (1.0-pow(Nm,lam))/(pi*lam)\
            if lam < -1 else -np.log(Nm)/pi
    K = np.size(h);  N = np.size(ImgRow)
    X = np.zeros(K); AImgRow = np.zeros(N)

    gp = np.gradient (ImgRow)
    for n in range(0,N-1):
        Nm = (N-n)/(N-n-1.0)
        for k in range(0,K):
            X[k] = pow(Nm,lam[k])*X[k] + h[k]*Gamma(Nm,lam[k])*gp[n]
        AImgRow[n] = X.sum()

    AImgRow[N-1] = AImgRow[N-2]  # special case for N=N-1
    return -AImgRow

def HansenLaw (data,mask=0b1111,calc_speeds=True,verbose=True):
    """ Split image into quadrants and co-add
        mask =   Q3 Q2 Q1 Q0
                ( Q1 | Q0)
                 --------
                ( Q2 | Q3)
    """  
    (N,M)=np.shape(data)
    N2 = N//2
    if verbose:
        print ("HL: Calculating inverse Abel transform: image size {:d}x{:d}".format(N,M))

# split image into quadrants
    t0=time()
    left,right = np.array_split(data,2,axis=1)  # (left | right)  half image
    Q0,Q3 = np.array_split(right,2,axis=0)      # top/bottom of right half
    Q1,Q2 = np.array_split(left,2,axis=0)       # top/bottom of left half
    Q0 = np.fliplr(Q0)                          # reorientate
    Q2 = np.flipud(Q2)
    Q3 = np.fliplr(np.flipud(Q3))

# combine selected quadrants into one or loop through if none 
    if mask>0:
        if verbose:
            print ("HL: co-adding quadrants mask={:s}".format(bin(mask)))

        Q = Q0*(mask&(1<<0))+Q1*(mask&(1<<1))+Q2*(mask&(1<<2))+Q3*(mask&(1<<3))

        # inverse Abel transform of combined quadrant, applied to each row
        AQ=[HansenLaw_transform(Q[row]) for row in range(N2)]

        # reform image
        Top    = np.concatenate ((AQ,np.fliplr(AQ)),axis=1)
        Bottom = np.flipud(Top)
        recon  = np.concatenate ((Top,Bottom),axis=0)

    else:
        if verbose: print ("HL: individual quadrants")

        # Fix me! - there should be a more python way
        # inversion each quandrante, one row at a time
        AQ0 = [HansenLaw_transform(Q0[row]) for row in range(N2)]
        AQ1 = [HansenLaw_transform(Q1[row]) for row in range(N2)]
        AQ2 = [HansenLaw_transform(Q2[row]) for row in range(N2)]
        AQ3 = [HansenLaw_transform(Q3[row]) for row in range(N2)]

        # reform image
        Top    = np.concatenate ((AQ1,np.fliplr(AQ0)),axis=1)
        Bottom = np.flipud(np.concatenate ((AQ2,np.fliplr(AQ3)),axis=1))
        recon  = np.concatenate ((Top,Bottom),axis=0)
            
    if verbose:
        print ("HL: Inverse Abel transform in {:.2f} seconds".format(time()-t0))
    if calc_speeds:
        if verbose:
            print('Generating speed distribution...')
            t1 = time()

        speeds = calculate_speeds(recon, N)

        if verbose: print('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon
