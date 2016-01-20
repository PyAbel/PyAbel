# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from time import time
from math import exp, log, pow, pi
from abel.tools.vmi import calculate_speeds
from abel.tools.symmetry import  get_image_quadrants, put_image_quadrants

###############################################################################
# hansenlaw - a recursive method forward/inverse Abel transform algorithm 
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
    Hansen and Law J. Opt. Soc. Am. A 2, 510-520 (1985).::
    
                      
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
    toward the image centre (origin). i.e. when n=N-1, R=Rmax, and
    when n=0, R=0. This fits well with processing the image one 
    quadrant (chosen orientation to be rightside-top), or one right-half 
    image at a time.

    Use (f/i)abel_transform (IM) to transform a whole image
       
    Parameters
    ----------
    IM : 2D np.array
        One quadrant (or half) of the image oriented top-right::
    
             +--------      +--------+              |
             |      *       | *      |              |
             |   *          |    *   |  <----------/
             |  *           |     *  |         
             +--------      o--------+
             |  *           |     *  |
             |   *          |    *   |
             |     *        | *      |
             +--------      +--------+
                          
             Image centre `o' should be within a pixel 
             (i.e. an odd number of columns)
             [Use abel.tools.vmi.find_image_center_by_slice () to transform] 

    dr : float
        Sampling size (=1 for pixel images), used for Jacobian scaling

    inverse: boolean 
        forward (False) or inverse (True) Abel transform

    Returns
    -------
    AIM : 2D np.array
        forward/inverse Abel transform image 

    """

_hansenlaw_docstring = \
    """ 
    inverse Abel transform image

    options to exploit image symmetry 
           - select quadrants
           - combine quadrantsto improve signal

    Parameters
    ----------
    IM: 2D np.array
        Image data shape (rows, cols)

    dr : float 
        radial sampling size (=1 for pixel images), used to scale result

    inverse: boolean 
        forward (False) or inverse (True) Abel transform

    use_quadrants: boolean tuple (Q0,Q1,Q2,Q3)
        select quadrants to be used in the analysis::

             +--------+--------+                
             | Q1   * | *   Q0 |
             |   *    |    *   |                               
             |  *     |     *  |                               AQ1 | AQ0
             +--------o--------+ --(inverse Abel transform)--> ----o----
             |  *     |     *  |                               AQ2 | AQ3 
             |   *    |    *   |
             | Q2  *  | *   Q3 |          AQi == inverse Abel transform  
             +--------+--------+                 of quadrant Qi
 
       ::

       (1) vertical_symmetry = True 
       ::
 
           Combine:  `Q01 = Q1 + Q2, Q23 = Q2 + Q3`
           inverse image   AQ01 | AQ01     
                           -----o-----            
                           AQ23 | AQ23
       ::

       (2) horizontal_symmetry = True
       ::

           Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
           inverse image   AQ12 | AQ03       
                           -----o-----
                           AQ12 | AQ03
       ::
 
       (3) vertical_symmetry = True, horizontal = True
       :: 
        
           Combine: Q = Q0 + Q1 + Q2 + Q3
           inverse image   AQ | AQ       
                           ---o---  all quadrants equivalent
                           AQ | AQ
 

    verbose: boolean
        verbose output, timings etc.

    """  


# functions to conform to naming conventions: contributing.md ------------

def fabel_hansenlaw_transform(IM, dr=1):
    """
    Forward Abel transform for one-quadrant
    """
    return _abel_hansenlaw_transform_core(IM, dr=dr, inverse=False)


def iabel_hansenlaw_transform(IM, dr=1):
    """
    Inverse Abel transform for one-quadrant
    """
    return _abel_hansenlaw_transform_core(IM, dr=dr, inverse=True)


def fabel_hansenlaw(IM, dr=1, **args):
    """
    Helper function - splits image into quadrants for processing by
    fabel_hansenlaw_transform
    """
    return _abel_hansenlaw_core(IM, dr=dr, inverse=False, **args)


def iabel_hansenlaw(IM, dr=1, **args):
    """
    Helper function - splits image into quadrants for processing by
    iabel_hansenlaw_transform
    """
    return _abel_hansenlaw_core(IM, dr=dr, inverse=True, **args) 

# ----- end naming ---------------


def _abel_hansenlaw_transform_core(IM, dr=1, inverse=False):
    """
    Hansen and Law JOSA A2 510 (1985) forward and inverse Abel transform
    for right half (or right-top quadrant) of an image.
    """

    IM = np.atleast_2d(IM)  
    N = np.shape(IM)         # shape of input quadrant (half) 
    AIM = np.zeros(N)        # forward/inverse Abel transform image

    rows, cols = N 

    # constants listed in Table 1.
    h = [0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3]
    lam = [0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1]

    K = np.size(h)
    X = np.zeros((rows, K))

    # Two alternative Gamma functions for forward/inverse transform
    # Eq. (16c) used for the forward transform           
    def fgamma(Nm, lam, n):   
        return 2*n*(1-pow(Nm, (lam+1)))/(lam+1)

    # Eq. (18) used for the inverse transform           
    def igamma(Nm, lam, n):   
        if lam < -1:
            return (1-pow(Nm, lam))/(pi*lam)
        else:
            return -np.log(Nm)/pi    

    if inverse:   # inverse transform
        gamma = igamma 
        # g' - derivative of the intensity profile
        if rows > 1:
            gp = np.gradient(IM)[1]    # second element is gradient along 
                                       # the columns
        else:  # If there is only one row
            gp = np.atleast_2d(np.gradient(IM[0]))
    else:  # forward transform
        gamma = fgamma
        gp = IM   

    # ------ The Hansen and Law algorithm ------------
    # iterate along columns, starting outer edge (right side) 
    # toward the image center

    for n in range(cols-2, 0, -1):       
        Nm = (n+1)/n          # R0/R
        
        for k in range(K):  # Iterate over k, the eigenvectors?
            X[:, k] = pow(Nm, lam[k])*X[:, k] +\
                     h[k]*gamma(Nm, lam[k], n)*gp[:, n]  # Eq. (15 or 17)            
        AIM[:, n+1] = X.sum(axis=1)

    # special case for the end pixel
    AIM[:, 0] = AIM[:, 1]  

    # for some reason shift by 1 pixel aligns better? - FIX ME!
    #if inverse:
    #    AIM = np.c_[AIM[:, 1:],AIM[:, -1]]

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    if inverse:
        return AIM*np.pi/dr    # 1/dr - from derivative
    else:
        return -AIM*np.pi*dr   # forward still needs '-' sign

    # ---- end abel_hansenlaw_transform ----


def _abel_hansenlaw_core(IM, dr=1, inverse=True, 
                         use_quadrants=(True, True, True, True), 
                         vertical_symmetry=False, horizontal_symmetry=False, 
                         verbose=False):
    """
    Returns the forward or the inverse Abel transform of a function
    using the Hansen and Law algorithm

    """

    verboseprint = print if verbose else lambda *a, **k: None
    
    if IM.ndim == 1 or np.shape(IM)[0] <= 2:
            raise ValueError('Data must be 2-dimensional.'
                             'To transform a single row, use'
                             'iabel_hansenlaw_transform().')

    rows, cols = np.shape(IM)

    if not np.any(use_quadrants):
        verboseprint("HL: Error: no image quadrants selected to use")
        return np.zeros((rows, cols))
        
    verboseprint("HL: Calculating inverse Abel transform:",
                 " image size {:d}x{:d}".format(rows, cols))

    t0 = time()
    
    # split image into quadrants
    Q0, Q1, Q2, Q3 = get_image_quadrants(IM, reorient=True,
                         vertical_symmetry=vertical_symmetry,
                         horizontal_symmetry=horizontal_symmetry)

    verboseprint("HL: Calculating inverse Abel transform ... ")

    # HL inverse Abel transform for quadrant 1
    # all possibilities include Q1
    AQ1 = _abel_hansenlaw_transform_core(Q1, dr, inverse) 

    if vertical_symmetry:
        AQ2 = _abel_hansenlaw_transform_core(Q2, dr, inverse)

    if horizontal_symmetry:
        AQ0 = _abel_hansenlaw_transform_core(Q0, dr, inverse)

    if not vertical_symmetry and not horizontal_symmetry:
        AQ0 = _abel_hansenlaw_transform_core(Q0, dr, inverse)
        AQ2 = _abel_hansenlaw_transform_core(Q2, dr, inverse)
        AQ3 = _abel_hansenlaw_transform_core(Q3, dr, inverse)

    # reassemble image
    recon = put_image_quadrants((AQ0, AQ1, AQ2, AQ3), odd_size=cols % 2,
                                vertical_symmetry=vertical_symmetry,
                                horizontal_symmetry=horizontal_symmetry)

    verboseprint("{:.2f} seconds".format(time()-t0))

    return recon


# append the same docstring to all functions - borrowed from @rth
iabel_hansenlaw_transform.__doc__ += _hansenlaw_header_docstring +\
                                     _hansenlaw_transform_docstring
fabel_hansenlaw_transform.__doc__ += _hansenlaw_header_docstring +\
                                     _hansenlaw_transform_docstring
iabel_hansenlaw.__doc__ += _hansenlaw_header_docstring + _hansenlaw_docstring
fabel_hansenlaw.__doc__ += _hansenlaw_header_docstring +\
                           _hansenlaw_docstring.replace('AQ', 'fQ')\
                           .replace('(inverse', '(forward')\
                           .replace('== inverse', '== forward')\
                           .replace('inverse image', 'forward image')
#_abel_hansenlaw_transform_core.__doc__ += _hansenlaw_header_docstring +\
#                                          _hansenlaw_transform_docstring
