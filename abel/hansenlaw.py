# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

#from numba import jit
import numpy as np
from time import time
from math import exp, log, pow, pi
from abel.tools import calculate_speeds, add_image_col, delete_image_col

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

def iabel_hansenlaw_transform(IM):
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

        Evaluation via Eqs. (14), (17) & (18).

        Recursion method proceeds from the outer edge of the image
        toward the image centre (origin). i.e. when n=0, R=Rmax, and
        when n=N-1, R=0. This fits well with processing the image one 
        half at a time (or one quandrant to exploit image symmetry).

        Parameters:
        ----------
         - IM: a rows x cols numpy array = one half-side of the image
                                           oriented centre on right side
                 +--------+      --------+ 
                 |      * |       *      |
                 |   *    |          *   |
                 |  *     |           *  |
                 |        +              |
                 |  *     |           *  |
                 |   *    |          *   |
                 |     *  |       *      |
                 +--------+      --------+
            
    """

    N    = np.shape(IM)  # quadrant size 
    AImg = np.zeros(N)   # inverse Abel transformed IM array
    
    nrows,ncols = N      # number of rows, number of columns

    # constants listed in Table 1.
    h   = [0.318,0.19,0.35,0.82,1.8,3.9,8.3,19.6,48.3]
    lam = [0.0,-2.1,-6.2,-22.4,-92.5,-414.5,-1889.4,-8990.9,-47391.1]

    # Eq. (18)            
    def Gamma (Nm,lam): 
        if lam < -1:
            return (1-pow(Nm,lam))/(pi*lam)
        else:
            return -np.log(Nm)/pi    

    K = np.size(h)
    X = np.zeros((nrows,K))

    # g' - derivative of the intensity profile

    gp = np.gradient(IM)[1]  # take the second element which is the gradient along the columns

    # iterate along the column, starting at the outer edge to the image center
    for col in range(ncols-1):       
        Nm = (ncols-col)/(ncols-col-1)    # R0/R 
        
        for k in range(K): # Iterate over k, the eigenvectors
            X[:,k] = pow(Nm,lam[k])*X[:,k] + h[k]*Gamma(Nm,lam[k])*gp[:,col] # Eq. (17)            
            
        AImg[:,col] = X.sum(axis=1)

    AImg[ncols-1] = AImg[ncols-2]  # special case for the center pixel
    
    return -AImg


def iabel_hansenlaw (data,calc_speeds=True,verbose=True,quad=(False,False,False,False)):
    """ Helper function for Hansen Law inverse Abel transform.
        (0) add centre column to odd size width images 
        (1) split image to 2 halves
        (2) inverse Abel transform of half (iabel_hansenlaw_transform)
            (option) inverse Abel transform of combined quadrants
        (3) reform image
        (4) (optionally) calculate the radial integration of the image (calc_speeds)

        Parameters:
        ----------
         - data: a NxM numpy array
                 image is inverted one half-side at a time

                    left      right
                 +---------+---------+                
                 |       * | *       |
                 |   *     |     *   |  
                 |  *      |      *  |
                 |         +         |
                 |  *      |      *  |
                 |   *     |     *   |
                 |      *  |  *      |
                 +---------+---------+

         - calc_speeds: boolean, evaluate speed profile
         - verbose: boolean, more output, timings etc.
         - quad: boolean tuple, (Q0,Q1,Q2,Q3) 
                 exploit image symmerty combining selected Qi quadrants into one
                 for transform:

                   +----+----+    +---+                  +----+----+
                   | Q1 | Q0 |    | Q |                  | AQ | AQ |
                   +----+----+ -> +---+  -(transform) -> +----+----+
                   | Q2 | Q3 |                           | AQ | AQ |
                   +----+----+                           +----+----+

                 where Q = Q0+Q1+Q2+Q3 for quad = (True,True,True,True)
                           Q0+Q1+Q2                True,True,True,False
                           etc
    """  
    verboseprint = print if verbose else lambda *a, **k: None

    (nrows,ncols) = data.shape
    verboseprint ("HL: Calculating inverse Abel transform:",
                      " image size {:d}x{:d}".format(nrows,ncols))
   
    # (1) Image width even? -------------------
    #      Hansen and Law algorithm works with image split in half, an even size image

    # add a centre column if needed to make image width even

    oddimage = ncols%2    # whether image has an odd pixel number width
    if oddimage:
        verboseprint("HL: odd size image, add centre row + column")
        data = add_image_col (data)

    t0=time()
    
    # (2) Split image into half -----------------------------
    Hleft, Hright = left_right_image (data)     # Hleft | Hright

    # (option) Symmetry - combine quadrants to improve signal
    #                     split into quadrants

    if np.any(quad):                           #    Q1 | Q0  
        Q1, Q2 = top_bottom_image (Hleft)      #   ----+----
        Q0, Q3 = top_bottom_image (Hright)     #    Q2 | Q3  
        # orient the same as Q1
        Q0 = np.fliplr(Q0)
        Q2 = np.flipud(Q2)
        Q3 = np.flipud(np.fliplr(Q3))
        # combine into a single quadrant
        Qcomb = Q0*quad[0]+Q1*quad[1]+Q2*quad[2]+Q3*quad[3]
   
    # (3) Hansen and Law inverse Abel transform 

    verboseprint ("HL: Calculating inverse Abel transform: ... ")

    if np.any(quad):                                                        
        # quadrants                                                          #  AQ
        AQcomb = iabel_hansenlaw_transform(Qcomb)                            # ----
        AHleft = AHright = np.concatenate((AQcomb,np.flipud(AQcomb)),axis=0) #  AQ
    else:
        # halves
        AHleft  = (iabel_hansenlaw_transform(Hleft))                #  AHl | AHr
        AHright = (iabel_hansenlaw_transform(np.fliplr(Hright)))  

    # (4) reform image ----------
    AHright = np.fliplr(AHright)

    recon = np.concatenate((AHleft,AHright),axis=1)

    verboseprint ("{:.2f} seconds".format(time()-t0))

    # (6) return image to the input shape -----------------
    if oddimage:
        verboseprint ("HL: return image to input shape")
        recon = delete_image_col (recon) 

    # (7) optionally calculate speed distribution -----------------
    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        # centre of even image is corner of pixel, not centre of pixel, issue #39
        # shift by 1/2 pixel.
        n2 = nrows//2
        m2 = ncols//2
        image_centre = (n2,m2) if oddimage else (n2-1/2,m2-1/2)
        speeds = calculate_speeds(recon,origin=image_centre)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon

def left_right_image (data):
    half   = data.shape[1]//2
    Hleft  = data[:,:half]
    Hright = data[:,half:]
    
    return Hleft, Hright

def top_bottom_image (Hdata):
    mid    = Hdata.shape[0]//2
    top    = Hdata[:mid]
    bottom = Hdata[mid:]

    return top, bottom
