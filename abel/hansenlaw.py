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

_hansenlaw_header_docstring = \
    """ 
    Forward/Inverse Abel transformation using the algorithm of: 
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

         f = iabel_hansenlaw(g) - inverse Abel transform of image g
         g = fabel_hansenlaw(f) - forward Abel transform of image f
        
         (f/i)abel_hansenlaw_transform() - core algorithm

    """

_hansenlaw_transform_docstring = \
    """

    Core Hansen and Law Abel transform

    Recursion method proceeds from the outer edge of the image
    toward the image centre (origin). i.e. when n=0, R=Rmax, and
    when n=N-1, R=0. This fits well with processing the image one 
    quadrant (oriented top left), or one left-half image at a time.

    Use (f/i)abel_transform (img) to transform a whole image
       
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

     -  dr: float - sampling size (=1 for pixel images), 
                    used for scaling result
     - inverse: boolean: False = forward Abel transform
                         True  = inverse Abel transform
    Return:
    -------
     - Aimg: a rows x cols numpy array, forward/inverse Abel transform image

    """

_hansenlaw_docstring = \
    """ 
    inverse Abel transform image

    options to exploit image symmetry 
           - select quadrants
           - combine quadrantsto improve signal

    Parameters:
    ----------
     - img: a rows x cols numpy array
     -  dr: float - sampling size (=1 for pixel images), used for scaling result
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
      - inverse: boolean: False = forward Abel transform
                          True  = inverse Abel transform

    """  


# functions to conform to naming conventions: contributing.md ------------

def fabel_hansenlaw_transform(img, dr=1):
    """
    Forward Abel transform for one-quadrant
    """
    return _abel_hansenlaw_transform_wrapper(img, dr=dr, inverse=False)

def iabel_hansenlaw_transform(img, dr=1):
    """
    Inverse Abel transform for one-quadrant
    """
    return _abel_hansenlaw_transform_wrapper(img, dr=dr, inverse=True)

def fabel_hansenlaw(img, dr=1, **args):
    """
    Helper function - splits image into quadrants for processing by
    fabel_hansenlaw_transform
    """
    return _abel_hansenlaw_wrapper(img, dr=dr, inverse=False, **args)

def iabel_hansenlaw(img, dr=1, **args):
    """
    Helper function - splits image into quadrants for processing by
    iabel_hansenlaw_transform
    """
    return _abel_hansenlaw_wrapper(img, dr=dr, inverse=True, **args) 

# ----- end naming ---------------

def _abel_hansenlaw_transform_wrapper(img, dr=1, inverse=False):
    """
    Hansen and Law JOSA A2 510 (1985) forward and inverse Abel transform
    for left half (or left-top quadrant) of an image.

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

    if inverse: # inverse transform
        Gamma = iGamma 
        # g' - derivative of the intensity profile
        if rows>1:
            gp = np.gradient(img)[1]  # second element is gradient along 
                                      # the columns
        else: # If there is only one row
            gp = np.atleast_2d(np.gradient(img[0]))
    else:  # forward transform
        Gamma = fGamma
        gp = img   

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
    
    if Aimg.shape[0] == 1:
        if inverse:
            return -Aimg[0]*np.pi/dr    # 1/dr - from derivative
        else:
            return -Aimg[0]*np.pi*dr
    else:
        if inverse:
            return -Aimg*np.pi/dr 
        else:
            return -Aimg*np.pi*dr 

    # ---- end abel_hansenlaw_transform ----


def _abel_hansenlaw_wrapper(img, dr=1, inverse=True, 
                            use_quadrants=(True,True,True,True), 
                            vertical_symmetry=False, horizontal_symmetry=False, 
                            calc_speeds=False, verbose=False):
    """
    Returns the forward or the inverse Abel transform of a function
    sampled using the Hansen and Law algorithm

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
        AQ0 = _abel_hansenlaw_transform_wrapper(Q0, dr, inverse)
        AQ1 = _abel_hansenlaw_transform_wrapper(Q1, dr, inverse)
        AQ2 = _abel_hansenlaw_transform_wrapper(Q2, dr, inverse)
        AQ3 = _abel_hansenlaw_transform_wrapper(Q3, dr, inverse)

    else:  # combine selected quadrants according to assumed symmetry
        if vertical_symmetry:   # co-add quadrants
            Q0 = Q1 = Q0*use_quadrants[0]+Q1*use_quadrants[1] 
            Q2 = Q3 = Q2*use_quadrants[2]+Q3*use_quadrants[3] 

        if horizontal_symmetry:
            Q1 = Q2 = Q1*use_quadrants[1]+Q2*use_quadrants[2] 
            Q0 = Q3 = Q0*use_quadrants[0]+Q3*use_quadrants[3] 

        # HL inverse Abel transform for quadrant 1
        AQ1 = _abel_hansenlaw_transform_wrapper(Q1, dr, inverse)  # all possibilities include Q1

        if vertical_symmetry:
            AQ0 = AQ1
            AQ3 = AQ2 = _abel_hansenlaw_transform_wrapper(Q2, dr, inverse)

        if horizontal_symmetry:
            AQ2 = AQ1
            AQ3 = AQ0 = _abel_hansenlaw_transform_wrapper(Q0, dr, inverse)

    # reassemble image
    recon = put_image_quadrants ((AQ0,AQ1,AQ2,AQ3), odd_size=cols%2)

    verboseprint ("{:.2f} seconds".format(time()-t0))

    if calc_speeds:
        verboseprint('Generating speed distribution ...')
        t1 = time()

        speeds = calculate_speeds(recon)

        verboseprint('{:.2f} seconds'.format(time() - t1))
        return recon, speeds
    else:
        return recon

# append the same docstring to all functions - borrowed from @rth
iabel_hansenlaw_transform.__doc__ += _hansenlaw_header_docstring +\
                                     _hansenlaw_transform_docstring
fabel_hansenlaw_transform.__doc__ += _hansenlaw_header_docstring +\
                                     _hansenlaw_transform_docstring
iabel_hansenlaw.__doc__ += _hansenlaw_header_docstring + _hansenlaw_docstring
fabel_hansenlaw.__doc__ += _hansenlaw_header_docstring +\
                           _hansenlaw_docstring.replace('AQ','fQ')\
                                      .replace('(inverse','(forward')\
                                      .replace('== inverse','== forward')\
                                      .replace('inverse image','forward image')
#_abel_hansenlaw_transform_wrapper.__doc__ += _hansenlaw_header_docstring +\
                                              #_hansenlaw_transform_docstring
