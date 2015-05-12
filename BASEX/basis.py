#!/usr/bin/pthon
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from math import exp, log

import numpy as np
from scipy.special import gammaln
import sys

MAX_OFFSET = 4000

def generate_basis_sets(n=1001, nbf=500, verbose=True):
    """ 
    Generate the basis set for the BASEX method. 

    This function was adapted from the a matlab script BASIS2.m,
    with some optimizations.

    Parameters:
    -----------
      n : integer : problem size ?
      nbf: integer: number of basis functions ?

    Returns:
    --------
      M, Mc : np.matrix
    """
    if n % 2 == 0:
        raise ValueError('The n parameter must be odd (more or less sure about it).')

    if n//2 < nbf:
        raise ValueError('The number of basis functions nbf cannot be larger then the number of points n!')


    Rm = n//2 + 1

    I = np.arange(1,n+1)

    R2 = (I - Rm)**2
    # R = I - Rm
    M = np.zeros((n, nbf))
    Mc = np.zeros((n, nbf))

    M[:,0] = 2*np.exp(-R2)
    Mc[:,0] = np.exp(-R2)

    gammaln_0o5 = gammaln(0.5) 

    if verbose:
        print('Generating BASEX basis sets for n = {}, nbf = {}:\n'.format(n, nbf))
        sys.stdout.write('0')
        sys.stdout.flush()

    # the number of elements used to calculate the projected coefficeints
    delta = np.fmax(np.arange(nbf)*32 - 4000, 4000) 
    for k in range(1, nbf):
        k2 = k*k # so we don't recalculate it all the time
        log_k2 = log(k2) 
        angn = exp(
                    k2-2*k2*log(k) +
                    #np.log(np.arange(0.5, k2 + 0.5)).sum() # original version
                    gammaln(k2 + 0.5) - gammaln_0o5  # optimized version
                    )
        M[Rm-1, k] =  2*angn

        for l in range(1, n-Rm+1):
            l2 = l*l
            log_l2 = log(l2)

            val = exp(k2 - l2 + 2*k2*log((1.0*l)/k))
            Mc[l-1+Rm, k] = val
            Mc[Rm-l-1, k] = val

            aux = val + angn*Mc[l+Rm-1, 0]

            p = np.arange(max(1, l2 - delta[k]), min(k2 - 1,  l2 + delta[k]) + 1)

            # We use here the fact that for p, k real and positive
            #
            #  np.log(np.arange(p, k)).sum() == gammaln(k) - gammaln(p) 
            #
            # where gammaln is scipy.misc.gammaln (i.e. the log of the Gamma function)
            #
            # The following line corresponds to the vectorized third
            # loop of the original BASIS2.m matlab file.


            aux += np.exp(k2 - l2 - k2*log_k2 + p*log_l2
                      + gammaln(k2+1) - gammaln(p+1) 
                      + gammaln(k2 - p + 0.5) - gammaln_0o5
                      - gammaln(k2 - p + 1)
                      ).sum()

            # End of vectorized third loop

            aux *= 2

            M[l+Rm-1, k] = aux
            M[Rm-l-1, k] = aux

        if verbose and k % 50 == 0:
            sys.stdout.write('...{}'.format(k))
            sys.stdout.flush()

    if verbose:
        print("...{}".format(k+1))


    return M, Mc

