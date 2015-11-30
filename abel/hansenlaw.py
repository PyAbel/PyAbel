# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from numba import jit
import numpy as np
import multiprocessing as mp 
from time import time
from math import exp, log, pow, pi
from abel.tools import calculate_speeds

###########################################################################
# hasenlaw - a recursive method inverse Abel transformation algorithm 
#
# Stephen Gibson - Australian National University, Australia
# Jason Gascooke - Flinders University, Australia
# 
# This algorithm is adapted by Jason Gascooke from the article
#   E. W. Hansen and P-L. Law
#  "Recursive methods for computing the Abel transform and its inverse"
#   J. Opt. Soc. Am A2, 510-520 (1985) doi: 10.1364/JOSAA.2.000510
#
#  J. R. Gascooke PhD Thesis:
#   "Energy Transfer in Polyatomic-Rare Gas Collisions and Van Der Waals 
#    Molecule Dissociation", Flinders University, 2000.
#
# Implemented in Python, with image quadrant co-adding, by Steve Gibson
#
###########################################################################

#@jit
def iabel_hansenlaw_transform (ImgRow):
    """ Inverse Abel transformation using the algorithm of: 
        Hansen and Law J. Opt. Soc. Am A2, 510-520 (1985).
                        
                       ∞                
                       ⌠                
                   -1  ⎮   g'(R)     
           f(r) =  ─── ⎮ ──────────── dR      Eq. (2a)
                    π  ⎮    _________   
                       ⎮   ╱  2    2    
                       ⎮ ╲╱  R  - r     
                       ⌡                
                       r                 

        where f(r) = reconstructed image (source) function
              g'(R) = derivative of the projection (measured) function

        Evaluation via Eqs. (14), (17) & (18).

        Recursion method proceeds from the outer edge of the image
        toward the image centre (origin). i.e. when n=0, R=Rmax, and
        when n=N-1, R=0. This fits well with processing an image one 
        quadrant at a time.

        Parameters:
        ----------
         - ImgRow: a N/2 numpy vector = one row of one quadrant of the image
           |       orientated top/left
           |     +--------+              --------+ 
           \=>   |      * |               *      |
                 |   *    |                  *   |
                 |  *     |                   *  |
                 +--------+              --------+
                               |  *     |     *  |
                               |   *    |    *   |
                               |     *  | *      |
                               +--------+--------+

    """

    N = np.size(ImgRow)     # length of pixel row, note in this case N = n/2
    AImgRow = np.zeros(N)   # the inverse Abel transformed pixel row

# constants listed in Table 1.
    h   = [0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3]
    lam = [0.0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1]

# Eq. (18)
    Gamma = lambda Nm, lam: (1.0-pow(Nm,lam))/(pi*lam)\
            if lam < -1 else -np.log(Nm)/pi         

    K = np.size(h)
    X = np.zeros(K)

# g' - derivative of the intensity profile
    gp = np.gradient (ImgRow)   

# iterate along the pixel row, starting at the outer edge to the image centre
    for n in range(N-1):       
        Nm = (N-n)/(N-n-1.0)    # R0/R 
        for k in range(K):
            X[k] = pow(Nm,lam[k])*X[k] + h[k]*Gamma(Nm,lam[k])*gp[n] # Eq. (17)
        AImgRow[n] = X.sum()

    AImgRow[N-1] = AImgRow[N-2]  # special case for N=N-1
    return -AImgRow


def iabel_hansenlaw (data,quad=(True,True,True,True),calc_speeds=True,verbose=True,freecpus=1):
    """ Helper function for Hansen Law inverse Abel transform.
        (1) split image into quadrants
            (optional) exploit symmetry and co-add selected quadrants together
        (2) inverse Abel transform of quadrant (iabel_hansenlaw_transform)
        (3) reassemble image
            for co-add all inverted quadrants are identical
        (4) (optionally) calculate the radial integration of the image (calc_speeds)

        Parameters:
        ----------
         - data: a NxN numpy array
         - quad: boolean tuple, (Q0,Q1,Q2,Q3)
                 image is inverted one quadrant at a time
                 +--------+--------+                
                 | Q1   * | *   Q0 |
                 |   *    |    *   |  
                 |  *     |     *  |
                 +--------+--------+
                 |  *     |     *  |
                 |   *    |    *   |
                 | Q2  *  | *   Q3 |
                 +--------+--------+

           NB may exploit image symmetry, all quadrants are equivalent, co-add

           (1) quad.any() = False
                (FALSE,FALSE,FALSE,FALSE) => inverse Abel transform for 
                                             each quadrant

               inverse image   AQ1 | AQ0     AQi == inverse Abel transform  
                               ---------            of quadrant Q0
                               AQ2 | AQ3

           (2) quad.any() = True   exploits image symmetry to improve signal
                sum True quadrants Q = Q0 + Q1 + Q2 + Q3  (True,True,True,True)
                             or    Q = Q0 + Q1 + Q2       (True,True,True,False)
                                   etc

                inverse image   AQ | AQ
                                -------
                                AQ | AQ

          - calc_speeds: boolean, evaluate speed profile
          - verbose: boolean, more output, timings etc.
          - freecpus: integer, parallel processing, use all cpus - freecpus (default 1)
    """  
    verboseprint = print if verbose else lambda *a, **k: None
    
# parallel processing set pool = mp.Pool(1) if any multiprocessor issues
    pool = mp.Pool(processes=mp.cpu_count()-freecpus) 

    (N,M)=np.shape(data)
    N2 = N//2
    verboseprint ("HL: Calculating inverse Abel transform: image size {:d}x{:d}".format(N,M))

# split image into quadrants
    t0=time()
    left,right = np.array_split(data,2,axis=1)  # (left | right)  half image
    Q0,Q3 = np.array_split(right,2,axis=0)      # top/bottom of right half
    Q1,Q2 = np.array_split(left,2,axis=0)       # top/bottom of left half
    Q0 = np.fliplr(Q0)                          # reorientate
    Q2 = np.flipud(Q2)
    Q3 = np.fliplr(np.flipud(Q3))

# combine selected quadrants into one or loop through if none 
    if np.any(quad):
        verboseprint ("HL: Co-adding quadrants")

        Q = Q0*quad[0]+Q1*quad[1]+Q2*quad[2]+Q3*quad[3]

        verboseprint ("HL: Calculating inverse Abel transform ... ")
        # inverse Abel transform of combined quadrant, applied to each row
        AQ0 = pool.map(iabel_hansenlaw_transform,[Q[row] for row in range(N2)])

        AQ3 = AQ2 = AQ1 = AQ0  # all quadrants the same

    else:
        verboseprint ("HL: Individual quadrants")

        # inversion of each quandrant, one row at a time
        verboseprint ("HL: Calculating inverse Abel transform ... ")
        AQ0 = pool.map(iabel_hansenlaw_transform,[Q0[row] for row in range(N2)])
        AQ1 = pool.map(iabel_hansenlaw_transform,[Q1[row] for row in range(N2)])
        AQ2 = pool.map(iabel_hansenlaw_transform,[Q2[row] for row in range(N2)])
        AQ3 = pool.map(iabel_hansenlaw_transform,[Q3[row] for row in range(N2)])

    # reform image
    Top    = np.concatenate ((AQ1,np.fliplr(AQ0)),axis=1)
    Bottom = np.flipud(np.concatenate ((AQ2,np.fliplr(AQ3)),axis=1))
    recon  = np.concatenate ((Top,Bottom),axis=0)
            
    verboseprint ("{:.2f} seconds".format(time()-t0))

    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        speeds = calculate_speeds(recon, N)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon
