# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from time import time
from math import exp, log, pow, pi
from abel.tools import calculate_speeds, get_image_quadrants,\
                       put_image_quadrants

################################################################################
# hasenlaw - a recursive method forwrd/inverse Abel transform algorithm 
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
# 2015-12-16: Modified to calculate the forward Abel transform
# 2015-12-03: Vectorization and code improvements Dan Hickstein and Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
################################################################################

# see fabel, iabel functions at bottom

def abel_hansenlaw_transform(img, forward=True):
    """ Forward/Inverse Abel transformation using the algorithm of: 
        Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985).
                        
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

        Evaluation via Eq. (15 or 17), using (16a), (16b), and (16c or 18)

        Recursion method proceeds from the outer edge of the image
        toward the image centre (origin). i.e. when n=0, R=Rmax, and
        when n=N-1, R=0. This fits well with processing the image one 
        quadrant (or half) at a time.

        Parameters:
        ----------
         - img: a rows x cols numpy array = one quadrant (or half) of the image
           |                                oriented top/left
           |     +--------+      --------+ 
           \=>   |      * |       *      |
                 |   *    |          *   |
                 |  *     |           *  |
                 +--------+      --------+
                 |  *     |           *  |
                 |   *    |          *   |
                 |     *  |       *      |
                 +--------+      --------+
                     |        
                     \=>            [ ]...[ ][ ]
                                     :     :  : 
                                    [ ]...[ ][ ]
         Image centre is mid-pixel  [ ]...[ ][+]
                                    [ ]...[ ][ ]
                                     :     :  : 
                                    [ ]...[ ][ ]
          - forward: boolean: True = forward Abel transform
                              False = inverse Abel transform
        Returns:
        --------
          - Aimg: a rows x cols numpy array, forward/inverse Abel transform image
    """
    img  = np.atleast_2d(img)  
    N    = np.shape(img)       # shape of input quadrant (half) 
    Aimg = np.zeros(N)         # forward/inverse Abel transform image

    rows,cols = N 

    # constants listed in Table 1.
    h   = [0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3]
    lam = [0.0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1]

    K = np.size(h)
    X = np.zeros((rows,K))

    # Two alternative Gamma functions for forward/inverse transform
    # Eq. (16c) used for the forward transform           
    def fGamma(Nm, lam, N, n):   
        Nn1 = N - n - 1
        return 2*Nn1*(1-pow(Nm,(lam+1)))/(lam+1)

    # Eq. (18) used for the inverse transform           
    def iGamma(Nm, lam, N, n):   
        if lam < -1:
            return (1.0-pow(Nm,lam))/(pi*lam)
        else:
            return -np.log(Nm)/pi    

    if forward: # forward transform
        Gamma = fGamma
        gp = img   
    else: # inverse transform
        # g' - derivative of the intensity profile
        Gamma = iGamma
        if rows>1:
            gp = np.gradient(img)[1]  # second element is gradient along the columns
        else: # If there is only one row
            gp = np.atleast_2d(np.gradient(img[0]))

    # ------ The Hansen and Law algorithm ------------
    # iterate along columns, starting outer edge (left side) toward image center

    for col in range(cols-1):       
        Nm = (cols-col)/(cols-col-1.0)    # R0/R 
        
        for k in range(K): # Iterate over k, the eigenvectors?
            X[:,k] = pow(Nm,lam[k])*X[:,k] +\
                     h[k]*Gamma(Nm,lam[k],cols,col)*gp[:,col] # Eq. (15 or 17)            
        Aimg[:,col] = X.sum(axis=1)

    # special case for the center pixel
    Aimg[:,cols-1] = Aimg[:,cols-2]  
    
    return -Aimg[::-1] if forward else -Aimg*np.pi  # flip(?) or scaling

    # ---- end abel_hansenlaw_transform ----


def abel_hansenlaw (img, use_quadrants=(True,True,True,True), 
                    vertical_symmetry=False, horizontal_symmetry=False, 
                    calc_speeds=False, verbose=False, forward=False):
    """ 
    Helper function abel_hansenlaw_transfor()
    Transforms the whole image
    
    Exploit image symmetry - select quadrants, combine to improve 
                             signal

    Parameters:
    ----------
     - img: a rows x cols numpy array
     - use_quadrants: boolean tuple, (Q0,Q1,Q2,Q3)
             +--------+--------+                
             | Q1   * | *   Q0 |
             |   *    |    *   |                               
             |  *     |     *  |                               AQ1 | AQ0
             +--------+--------+ --(inverse Abel transform)--> ---------
             |  *     |     *  |                               AQ2 | AQ3 
             |   *    |    *   |
             | Q2  *  | *   Q3 |          AQi == inverse Abel transform  
             +--------+--------+                 of quadrant Qi

       (1) vertical_symmetry = True

           Combine:  Q01 = Q1 + Q2, Q23 = Q2 + Q3
           inverse image   AQ01 | AQ01     
                           -----------            
                           AQ23 | AQ23

       (2) horizontal_symmetry = True

           Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
           inverse image   AQ12 | AQ03       
                           -----------
                           AQ12 | AQ03

       (3) vertical_symmetry = True, horizontal = True


           Combine: Q = Q0 + Q1 + Q2 + Q3
           inverse image   AQ | AQ       
                           -------  all quadrants equivalent
                           AQ | AQ


      - calc_speeds: boolean, evaluate speed profile
      - verbose: boolean, more output, timings etc.
      - forward: boolean: True = forward Abel transform
                          False = inverse Abel transform
    """  
    verboseprint = print if verbose else lambda *a, **k: None
    
    if img.ndim == 1 or np.shape(img)[0] <= 2:
            raise ValueError('Data must be 2-dimensional.'
                             'To transform a single row, use'
                             'iabel_hansenlaw_transform().')

    rows,cols = np.shape(img)

    if not np.any(use_quadrants):
        verboseprint ("HL: Error: no image quadrants selected to use")
        return np.zeros((rows,cols))
        
    verboseprint ("HL: Calculating inverse Abel transform:",
                      " image size {:d}x{:d}".format(rows,cols))

    t0=time()
    
    # split image into quadrants
    Q0,Q1,Q2,Q3 = get_image_quadrants(img, reorient=True)

    verboseprint ("HL: Calculating inverse Abel transform ... ")

    if not vertical_symmetry and not horizontal_symmetry\
                             and np.all(use_quadrants):
        # individual quadrant inverse Abel transform
        AQ0 = iabel_hansenlaw_transform(Q0)
        AQ1 = iabel_hansenlaw_transform(Q1)
        AQ2 = iabel_hansenlaw_transform(Q2)
        AQ3 = iabel_hansenlaw_transform(Q3)

    else:  # combine selected quadrants according to assumed symmetry
        if vertical_symmetry:   # co-add quadrants
            Q0 = Q1 = Q0*use_quadrants[0]+Q1*use_quadrants[1] 
            Q2 = Q3 = Q2*use_quadrants[2]+Q3*use_quadrants[3] 

        if horizontal_symmetry:
            Q1 = Q2 = Q1*use_quadrants[1]+Q2*use_quadrants[2] 
            Q0 = Q3 = Q0*use_quadrants[0]+Q3*use_quadrants[3] 

        # HL inverse Abel transform for quadrant 1
        AQ1 = iabel_hansenlaw_transform(Q1)  # all possibilities include Q1

        if vertical_symmetry:
            AQ0 = AQ1
            AQ3 = AQ2 = iabel_hansenlaw_transform(Q2)

        if horizontal_symmetry:
            AQ2 = AQ1
            AQ3 = AQ0 = iabel_hansenlaw_transform(Q0)

    # reassemble image
    recon = put_image_quadrants ((AQ0,AQ1,AQ2,AQ3),odd_size=True)

    verboseprint ("{:.2f} seconds".format(time()-t0))

    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        speeds = calculate_speeds(recon)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon

# functions to conform to naming conventions: contributing.md ------------

def fabel_hansenlaw_transform(img):
    return abel_hansenlaw_transform(img, forward=True)

def iabel_hansenlaw_transform(img):
    return abel_hansenlaw_transform(img, forward=False)

def fabel_hansenlaw(img, use_quadrants=(True,True,True,True), 
                    vertical_symmetry=False, horizontal_symmetry=False, 
                    calc_speeds=False, verbose=False):

    return abel_hansenlaw(img, use_quadrants, vertical_symmetry, 
                          horizontal_symmetry, calc_speeds, verbose,
                          forward=True)

def iabel_hansenlaw(img, use_quadrants=(True,True,True,True), 
                    vertical_symmetry=False, horizontal_symmetry=False, 
                    calc_speeds=False, verbose=False):

    return abel_hansenlaw(img, use_quadrants, vertical_symmetry, 
                          horizontal_symmetry, calc_speeds, verbose,
                          forward=False)
