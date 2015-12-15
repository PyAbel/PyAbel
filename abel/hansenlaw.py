# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from numba import jit
import numpy as np
from time import time
from math import exp, log, pow, pi
from abel.tools import calculate_speeds, get_image_quadrants,\
                       put_image_quadrants

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
# 2015-12-03: Vectorization and code improvements Dan Hickstein and Roman Yurchak
#             Previously the algorithm iterated over the rows of the image
#             now all of the rows are calculated simultaneously, which provides
#             the same result, but speeds up processing considerably.
###########################################################################

def iabel_hansenlaw_transform(img):
    """ Inverse Abel transformation using the algorithm of: 
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

        Evaluation via Eq. (17), using (16a), (16b), and (18)

        Recursion method proceeds from the outer edge of the image
        toward the image centre (origin). i.e. when n=0, R=Rmax, and
        when n=N-1, R=0. This fits well with processing the image one 
        quadrant (or half) at a time.

        Parameters:
        ----------
         - img: a rows x cols numpy array = one quadrant (or half) of the image
           |                               oriented top/left
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
        Returns:
        --------
          - Aimg: a rows x cols numpy array, inverse Abel transformed image
    """
    img  = np.atleast_2d(img)
    N    = np.shape(img)  # shape of quadrant (half) in this case N = n/2
    Aimg = np.zeros(N)    # inverse Abel transformed image

    nrows,ncols = N      # number of rows, number of columns

    # constants listed in Table 1.
    h   = [0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3]
    lam = [0.0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1]

    # Eq. (18)            
    def Gamma(Nm,lam):   
        if lam < -1:
            return (1.0-pow(Nm,lam))/(pi*lam)
        else:
            return -np.log(Nm)/pi    

    K = np.size(h)
    X = np.zeros((nrows,K))

    # g' - derivative of the intensity profile
    if nrows>1:
        gp = np.gradient(img)[1]  # second element is gradient along the columns
    else: # If there is only one row
        gp = np.atleast_2d(np.gradient(img[0]))

    # iterate along columns, starting outer edge (left side) toward image center
    for col in range(ncols-1):       
        Nm = (ncols-col)/(ncols-col-1.0)    # R0/R 
        
        for k in range(K): # Iterate over k, the eigenvectors?
            X[:,k] = pow(Nm,lam[k])*X[:,k] + h[k]*Gamma(Nm,lam[k])*gp[:,col] # Eq. (17)            
            
        Aimg[:,col] = X.sum(axis=1)

    Aimg[:,ncols-1] = Aimg[:,ncols-2]  # special case for the center pixel
    
    return -Aimg*np.pi     # scaling pi/delta(=1)
# ---- end iabel_hansenlaw_transform ----

def iabel_hansenlaw(img, calc_speeds=False, verbose=False):
    """
    Helper function: 
      - split image into two halves
      - inverse Abel transform
      - reassemble image
      - (optional) evaluate speeds
    """
    verboseprint = print if verbose else lambda *a, **k: None

    if img.ndim == 1 or np.shape(img)[0] <= 2:
        raise ValueError('Image must be 2-dimensional.'
                         ' To transform a single row'
                         'use iabel_hansenlaw_transform().')

    (rows,cols) = np.shape(img)
    if cols%2 != 1: 
        raise ValueError('Image size must be odd.',
                         'Use abel.tools.even2odd() to shift centre')
    midcol = cols//2
   
    verboseprint ("HL: Calculating inverse Abel transform:",
                      " image size {:d}x{:d}".format(rows,cols))

    t0=time()
    # split image into left-half, right-half
    Hleft  = img[:,:midcol+1]  # include centre-line column
    Hright = img[:,midcol:]    # with centre line, flipped

    Aleft  = iabel_hansenlaw_transform(Hleft)
    Aright = iabel_hansenlaw_transform(Hright[:,::-1])[:,::-1] # flipped

    # reassemble image removing duplicate centre column
    Aimg = np.concatenate ((Aleft,Aright[:,1:]),axis=1)

    verboseprint ("{:.2f} seconds".format(time()-t0))

    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        speeds = calculate_speeds(Aimg)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return Aimg, speeds
    else:
        return Aimg


def iabel_hansenlaw_symmetric (img, use_quadrants=(True,True,True,True), vertical_symmetry=True, horizontal_symmetry=True, calc_speeds=False, verbose=False):
    """ Helper function for Hansen Law inverse Abel transform.
        
        Exploit image symmetry - select quadrants, combine to improve 
                                 signal

        Parameters:
        ----------
         - img: a rowsXcols numpy array
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
    """  
    verboseprint = print if verbose else lambda *a, **k: None
    
    if img.ndim == 1 or np.shape(img)[0] <= 2:
            raise ValueError('Data must be 2-dimensional.'
                             'To transform a single row, use'
                             'iabel_hansenlaw_transform().')
        
    if not vertical_symmetry and not horizontal_symmetry and use_quadrants.all():
        return iabel_hansenlaw (img,calc_speeds,verbose)   # default routine

    (rows,cols) = np.shape(img)

    verboseprint ("HL: Calculating inverse Abel transform:",
                      " image size {:d}x{:d}".format(rows,cols))

    t0=time()
    
    # split image into quadrants
    Q0,Q1,Q2,Q3 = get_image_quadrants(img, reorient=True)

    if vertical_symmetry:   # co-add quadrants
        Q0 = Q1 = Q0*use_quadrants[0]+Q1*use_quadrants[1] 
        Q2 = Q3 = Q2*use_quadrants[2]+Q3*use_quadrants[3] 

    if horizontal_symmetry:
        Q1 = Q2 = Q1*use_quadrants[1]+Q2*use_quadrants[2] 
        Q0 = Q3 = Q0*use_quadrants[0]+Q3*use_quadrants[3] 

    verboseprint ("HL: Calculating inverse Abel transform ... ")
    
    # HL inverse Abel transform for quadrant 0
    AQ1 = iabel_hansenlaw_transform(Q1)  # all possibilities include Q1

    if vertical_symmetry:
        AQ0 = AQ1
        AQ3 = AQ2 = iabel_hansenlaw_transform(Q2)

    if horizontal_symmetry:
        AQ2 = AQ1
        AQ3 = AQ0 = iabel_hansenlaw_transform(Q0)

    # reassemble image
    recon = put_image_quadrants ((AQ0,AQ1,AQ2,AQ3))

    verboseprint ("{:.2f} seconds".format(time()-t0))

    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        speeds = calculate_speeds(recon)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon
