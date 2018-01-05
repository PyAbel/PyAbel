# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np

##############################################################################
#
# Abel analytical function transform pairs:  source <-> projection
#
# 04-Jan-2018 Dan Hickstein - code improvements
# 20-Dec-2017 Stephen Gibson - adapted code for PyAbel
# 20-Nov-2015 Dhrubajyoti Das - python gist
#             https://github.com/PyAbel/PyAbel/issues/19#issuecomment-158244527
#
# Note: preferable to call these functions via the class method:
#        func = abel.tools.analytical.TransformPair(n, profile=#)
#   see abel/tools/analytical.py for Class attributes
#
##############################################################################

_transform_pairs_docstring = \
r"""Analytical function Abel transform pairs

    profiles 1-7, table 1 of:
     `G. C.-Y Chan and G. M. Hieftje Spectrochimica Acta B 61, 31-41 (2006)
     <http://doi:10.1016/j.sab.2005.11.009>`_

    Note: profile4 does not produce a correct Abel transform pair due
          to typographical errors in the publications

    profile 8, curve B in table 2 of:
     `Hansen and Law J. Opt. Soc. Am. A 2 510-520 (1985)
     <http://doi:10.1364/JOSAA.2.000510>`_

    Note: the transform pair functions are more conveniently accessed via
      the class::

         func = abel.tools.analytical.TransformPair(n, profile=nprofile)

      which sets the radial range r and provides attributes:
          ``.func`` (source), ``.abel`` (projection), ``.r`` (radial range),
          ``.dr`` (step), ``.label`` (the profile name)


    Parameters
    ----------
    r : floats or numpy 1D array of floats
       value or grid to evaluate the function pair: ``0 < r < 1``

    Returns
    -------
    source, projection : tuple of 1D numpy arrays of shape `r`
        source function profile (inverse Abel transform of projection),
        projection functon profile (forward Abel transform of source)

"""


def a(n, r):
    """ coefficient

        .. math:: a_n = \sqrt{n^2 - r^2}

    """

    return np.sqrt(n*n - r*r)


def profile1(r):
    """**profile1**:
    `Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

     .. math::

          \epsilon(r) &= 0.75 + 12r^2 -32r^3  & 0 \le r \le 0.25

          \epsilon(r) &= \\frac{16}{27}(1 + 6r -15r^2 +8r^3) & 0.25 \lt r \le 1

          I(r) &= \\frac{1}{108}(128a_1 +a_{0.25}) + \\frac{2}{27}r^2
                    (283a_{0.25} - 112a_1) +

          & \,\,\,\, \\frac{8}{9}r^2\left[4(1+r^2)\ln\\frac{1+a_1}{r} -
            (4+31r^2)\ln\\frac{0.25+a_{0.25}}{r}\\right] &  0 \le r \le 0.25

          I(r) &= \\frac{32}{27}\left[a_1 - 7a_1 r + 3r^2(1+r^2)
                  \ln\\frac{1+a_1}{r}\\right]  & 0.25 \lt r \le 1

     ::

                           profile1
                  source                projection
              │                      │ o               
              │                      o  o              
              │    x                 │    o            
              │  x  x                │     o           
              │ x                    │                 
              x       x              │       o         
              │        x             │                 
              │                      │        o        
              │         x            │                 
              │                      │         o       
            ──┼───────────x─────   ──┼───────────o─────
              │                      │                 

    """

    if np.any(r <= 0) or np.any(r > 1):
        raise ValueError('r must be 0 < r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # r <= 0.25
    rm = r[r <= 0.25]

    # source
    em = 3/4 + 12*rm**2 - 32*rm**3

    # projection
    a4m = a(0.25, rm)
    a1m = a(1, rm)
    rm2 = rm**2
    Im = (128*a1m + a4m)/108 + (283*a4m - 112*a1m)*rm2*2/27 +\
         (4*(1 + rm2)*np.log((1 + a1m)/rm) -
          (4 + 31*rm2)*np.log((0.25 + a4m)/rm))*rm2*8/9

    # r > 0.25
    rp = r[r > 0.25]

    # source
    ep = (16/27)*(1 + 6*rp - 15*rp**2 + 8*rp**3)

    # projection
    a1p = a(1, rp)
    rp2 = rp**2
    Ip = (a1p - 7*a1p*rp2 + 3*rp2*(1 + rp2)*np.log((1 + a1p)/rp))*32/27

    source = np.concatenate((em, ep))
    proj = np.concatenate((Im, Ip))

    return source, proj


def profile2(r):
    """**profile2**:
    `Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

     .. math::

       \epsilon(r) &= 1 - 3r^2 + 2r^3 & 0 \le r \le 1

       I(r) &= a_1\left(1-\\frac{5}{2}r^2\\right) + 
               \\frac{3}{2}r^4\ln\\frac{1+a_1}{r} & 0 \le r \le 1

     ::

                           profile2
                  source                projection
              │                      │                 
              x x                    o o               
              │  x                   │  o              
              │    x                 │    o            
              │     x                │                 
              │                      │     o           
              │       x              │                 
              │                      │       o         
              │        x             │        o        
              │         x            │                 
            ──┼───────────x─────   ──┼─────────o───────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = 1 - 3*r*r + 2*r**3
    a1 = a(1, r)
    proj = a1*(1 - r**2*5/2) + r**4*np.log((1 + a1)/r)*3/2

    return source, proj


def profile3(r):
    """**profile3**:
    `Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

     .. math::

        \epsilon(r) &= 1-2r^2  & 0 \le r \le 0.5

        \epsilon(r) &= 2(1-r^2)^2 & 0.5 \lt r \le 1

        I(r) &= \\frac{4a_1}{3}(1+2r^2)-\\frac{2 a_{0.5}}{3}(1+8r^2) -
                4r^2\ln\\frac{1-a_1}{0.5+a_{0.5}} & 0 \le r \le 0.5

        I(r) &= \\frac{4a_1}{3}(1+2r^2)-4r^2\ln\\frac{1-a_1}{r} & 
                0.5 \lt r \le 1


     ::

                           profile3
                  source                projection
              │                      │                 
              x xx                   o o               
              │                      │  o              
              │    x                 │    o            
              │     x                │                 
              │                      │     o           
              │       x              │                 
              │                      │       o         
              │        x             │                 
              │                      │        o        
            ──┼─────────x───────   ──┼─────────o───────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # r <= 0.5
    rm = r[r <= 0.5]

    em = 1 - 2*rm**2

    a5m = a(0.5, rm)
    a1m = a(1, rm)
    # power rm**2 typo in Cremers
    Im = (4/3)*a1m*(1 + 2*rm**2) - (2/3)*a5m*(1 + 8*rm**2) -\
         4*rm**2*np.log((1 + a1m)/(0.5 + a5m))

    # r > 0.5
    rp = r[r > 0.5]
    a1p = a(1, rp)

    ep = 2*(1 - rp)**2
    Ip = (4/3)*a1p*(1 + 2*rp**2) - 4*rp**2*np.log((1 + a1p)/rp)

    source = np.concatenate((em, ep))
    proj = np.concatenate((Im, Ip))

    return source, proj


def profile4(r):
    """**profile4**: `Alvarez, Rodero, Quintero Spectochim. Acta B 57, 1665-1680 (2002) <https://doi.org/10.1016/S0584-8547(02)00087-3>`_

    WARNING: function pair incorrect due to typo errors in Table 1.

     .. math::

         \epsilon(r) &= 0.1 + 5.5r^2 - 5.25r^3 & 0 \le r \le 0.7

         \epsilon(r) &= -40.74 + 155.56r - 188.89r^2 + 74.07r^3 & 0.7 \lt r \le1

         I(r) &= 22.68862a_{0.7} - 14.811667a_1 + (217.557a_{0.7} -
                 193.30083a_1)r^2 + 

           & \,\,\, 155.56r^2\ln\\frac{1 + a_1}{0.7 + a_{0.7}} + 
             r^4\left(55.5525\ln\\frac{1 + a_1}{r} - 59.49\ln\\frac{0.7 + 
             a_{0.7}}{r}\\right)  & 0 \le r \le 0.7

         I(r) &= -14.811667a_1 - 193.0083a_1 r^2 + r^2(155.56 + 55.5525r^2)
                 \ln\\frac{1 + a_1}{r} & 0.7 \lt r \le 1


 ::

                           profile4
                  source                projection
              │                      │        oo       
              │                      │       o         
              │                      │     o     o     
              │                      │    o            
              │                      │            o    
              │                      │ oo              
              │        xx            o                 
              │       x   x          │                 
              │     x                │                 
              │    x                 │                 
            ──┼──x─────────x────   ──┼─────────────────
              │                      │                 

    """

    if np.any(r <= 0) or np.any(r > 1):
        raise ValueError('r must be 0 < r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # r <= 0.7
    rm = r[r <= 0.7]

    em = 0.1 + 5.51*rm**2 - 5.25*rm**3

    a7m = a(0.7, rm)
    a1m = a(1, rm)
    Im = 22.68862*a7m - 14.811667*a1m + (217.557*a7m - 193.30083*a1m)*rm**2 +\
         155.56*rm**2*np.log((1 + a1m)/(0.7 + a7m)) +\
         rm**4*(55.5525*np.log((1 + a1m)/rm) - 59.49*np.log((0.7 + a7m)/rm))

    # r > 0.7
    rp = r[r > 0.7]
    ep = -40.74 + 155.6*rp - 188.89*rp**2 + 74.07*rp**3
    a1p = a(1, rp)

    Ip = -14.811667*a1p - 193.0083*a1p*rp**2 +\
         rp**2*(155.56 + 55.5525*rp**2)*np.log((1 + a1p)/rp)

    source = np.concatenate((em, ep))
    proj = np.concatenate((Im, Ip))

    return source, proj


def profile5(r):
    """**profile5**: `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996) <https://doi.org/10.1016/j.amc.2014.03.043>`_

     .. math::

      \epsilon(r) &= 1 & 0 \le r \le 1

      I(r) &= 2a_1 & 0 \le r \le 1

 ::

                           profile5
                  source                projection
              │                      o oo              
              │                      │    oo           
              │                      │       oo        
              │                      │         o       
              │                      │                 
              │                      │           o     
              x xx xx xxx xx x       │                 
              │                      │            o    
              │                      │                 
              │                      │                 
            ──┼─────────────────   ──┼─────────────────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.ones_like(r)
    proj = 2*a(1, r)

    return source, proj


def profile6(r):
    """**profile6**: `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996) <https://doi.org/10.1016/j.amc.2014.03.043>`_

     .. math::

         \epsilon(r) &= (1-r^2)^{-\\frac{3}{2}} \exp\left[1.1^2\left(
                         1 - \\frac{1}{1-r^2}\\right)\\right] & 0 \le r \le 1

         I(r) &= \\frac{\sqrt{\pi}}{1.1a_1} \exp\left[1.1^2\left(
                         1 - \\frac{1}{1-r^2}\\right)\\right] & 0 \le r \le 1

 ::

                           profile6
                  source                projection
              │                      │                 
              │                      o oo              
              │                      │    oo           
              │                      │       o         
              │                      │                 
              x xx xx xx             │        o        
              │         x            │                 
              │                      │         o       
              │           x          │                 
              │                      │           o     
            ──┼─────────────────   ──┼─────────────────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.exp(1.1**2*(1 - 1/(1 - r**2)))/np.sqrt(1 - r**2)**3
    proj = np.exp(1.1**2*(1 - 1/(1 - r**2)))*np.sqrt(np.pi)/1.1/a(1, r)

    return source, proj


def profile7(r):
    """**profile7**:
    `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996)
    <https://doi.org/10.1016/j.amc.2014.03.043>`_

     .. math::

       \epsilon(r) &= \\frac{1}{2}(1+10r^2-23r^4+12r^6) & 0 \le r \le 1

       I(r) &= \\frac{8}{105}a_1(19 + 34r^2 - 125r^4 + 72r^6) & 0 \le r \le 1

     ::


                           profile7
                  source                projection
              │                      │                 
              │                      o oo oo           
              │                      │       o         
              │                      │                 
              │     x xx             │        o        
              │    x    x            │                 
              │                      │         o       
              │  x                   │                 
              x x         x          │                 
              │                      │           o     
            ──┼────────────x────   ──┼─────────────────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = (1 + 10*r**2 - 23*r**4 + 12*r**6)/2
    proj = a(1, r)*(19 + 34*r**2 - 125*r**4 + 72*r**6)*8/105

    return source, proj


def profile8(r):
    """**profile8**:
    Curve B table 2 of `Hansen and Law J. Opt. Soc. Am. A 2 510-520 (1985)
    <http://doi:10.1364/JOSAA.2.000510>`_

     .. math::

        \epsilon(r) &= (1-r^2)^{-\\frac{3}{2}}
                        \exp\left[\\frac{(1.1r)^2}{r^2-1}\\right]

        I(r) &= \\frac{\pi^\\frac{1}{2}}{1.1}(1-r^2)^{-\\frac{1}{2}}
                 \exp\left[\\frac{(1.1r)^2}{r^2-1}\\right]

    ::

                           profile8
                  source                projection
              │                      │                 
              │                      o oo              
              │                      │    oo           
              │                      │       o         
              │                      │                 
              x xx xx xx             │        o        
              │         x            │                 
              │                      │         o       
              │           x          │                 
              │                      │           o     
            ──┼─────────────────   ──┼─────────────────
              │                      │                 

    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.power(1-r**2, -3/2)*np.exp((1.1*r)**2/(r**2 - 1))
    proj = np.sqrt(np.pi)*np.power(1 - r**2, -1/2)*np.exp((1.1*r)**2\
                                                   / (r**2 - 1))/1.1

    return source, proj
