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
# Note: call these functions via the class method:
#   func = abel.tools.analytical.TransformPair(n, profile=#)
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

    (specific_profile_doc_info)

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


def a(n, x):
    return np.sqrt(n*n - x*x)


def profile1(r):

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

profile1.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile1:\n"
    "Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13): ::\n\n"
    "   .                   profile1\n"
    "   .          source                projection\n"
    "   .      │                      │ o               \n"
    "   .      │                      o  o              \n"
    "   .      │    x                 │    o            \n"
    "   .      │  x  x                │     o           \n"
    "   .      │ x                    │                 \n"
    "   .      x       x              │       o         \n"
    "   .      │        x             │                 \n"
    "   .      │                      │        o        \n"
    "   .      │         x            │                 \n"
    "   .      │                      │         o       \n"
    "   .    ──┼───────────x─────   ──┼───────────o─────\n"
    "   .      │                      │                 \n"
    "\n")


def profile2(r):

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = 1 - 3*r*r + 2*r**3
    a1 = a(1, r)
    proj = a1*(1 - r**2*5/2) + r**4*np.log((1 + a1)/r)*3/2

    return source, proj

profile2.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile2:\n"
    "Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13). ::\n\n"
    "   .                   profile2\n"
    "   .          source                projection\n"
    "   .      │                      │                 \n"
    "   .      x x                    o o               \n"
    "   .      │  x                   │  o              \n"
    "   .      │    x                 │    o            \n"
    "   .      │     x                │                 \n"
    "   .      │                      │     o           \n"
    "   .      │       x              │                 \n"
    "   .      │                      │       o         \n"
    "   .      │        x             │        o        \n"
    "   .      │         x            │                 \n"
    "   .    ──┼───────────x─────   ──┼─────────o───────\n"
    "   .      │                      │                 \n"
    "\n")


def profile3(r):

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

profile3.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile3:\n"
    "Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13). ::\n\n"
    "   .                   profile3\n"
    "   .          source                projection\n"
    "   .      │                      │                 \n"
    "   .      x xx                   o o               \n"
    "   .      │                      │  o              \n"
    "   .      │    x                 │    o            \n"
    "   .      │     x                │                 \n"
    "   .      │                      │     o           \n"
    "   .      │       x              │                 \n"
    "   .      │                      │       o         \n"
    "   .      │        x             │                 \n"
    "   .      │                      │        o        \n"
    "   .    ──┼─────────x───────   ──┼─────────o───────\n"
    "   .      │                      │                 \n"
    "\n")


def profile4(r):

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

profile4.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile4:\n"
    "Alvarez, Rodero, Quintero Spectochim. Acta B 57, 1665-1680 (2002)\n"
    "WARNING: function pair incorrect due to typo errors in Table 1. ::\n\n"
    "   .                   profile4\n"
    "   .          source                projection\n"
    "   .      │                      │        oo       \n"
    "   .      │                      │       o         \n"
    "   .      │                      │     o     o     \n"
    "   .      │                      │    o            \n"
    "   .      │                      │            o    \n"
    "   .      │                      │ oo              \n"
    "   .      │        xx            o                 \n"
    "   .      │       x   x          │                 \n"
    "   .      │     x                │                 \n"
    "   .      │    x                 │                 \n"
    "   .    ──┼──x─────────x────   ──┼─────────────────\n"
    "   .      │                      │                 \n"
    "\n")


def profile5(r):

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.ones_like(r)
    proj = 2*a(1, r)

    return source, proj

profile5.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile5:\n"
    "Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996). ::\n\n"
    "   .                   profile5\n"
    "   .          source                projection\n"
    "   .      │                      o oo              \n"
    "   .      │                      │    oo           \n"
    "   .      │                      │       oo        \n"
    "   .      │                      │         o       \n"
    "   .      │                      │                 \n"
    "   .      │                      │           o     \n"
    "   .      x xx xx xxx xx x       │                 \n"
    "   .      │                      │            o    \n"
    "   .      │                      │                 \n"
    "   .      │                      │                 \n"
    "   .    ──┼─────────────────   ──┼─────────────────\n"
    "   .      │                      │                 \n"
    "\n")


def profile6(r):

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.exp(1.1**2*(1 - 1/(1 - r**2)))/np.sqrt(1 - r**2)**3
    proj = np.exp(1.1**2*(1 - 1/(1 - r**2)))*np.sqrt(np.pi)/1.1/a(1, r)

    return source, proj

profile6.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile6:\n"
    "Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996). ::\n\n"
    "   .                   profile6\n"
    "   .          source                projection\n"
    "   .      │                      │                 \n"
    "   .      │                      o oo              \n"
    "   .      │                      │    oo           \n"
    "   .      │                      │       o         \n"
    "   .      │                      │                 \n"
    "   .      x xx xx xx             │        o        \n"
    "   .      │         x            │                 \n"
    "   .      │                      │         o       \n"
    "   .      │           x          │                 \n"
    "   .      │                      │           o     \n"
    "   .    ──┼─────────────────   ──┼─────────────────\n"
    "   .      │                      │                 \n"
    "\n")


def profile7(r):

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = (1 + 10*r**2 - 23*r**4 + 12*r**6)/2
    proj = a(1, r)*(19 + 34*r**2 - 125*r**4 + 72*r**6)*8/105

    return source, proj

profile7.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile7:\n"
    "Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996). ::\n\n"
    "   .                   profile7\n"
    "   .          source                projection\n"
    "   .      │                      │                 \n"
    "   .      │                      o oo oo           \n"
    "   .      │                      │       o         \n"
    "   .      │                      │                 \n"
    "   .      │     x xx             │        o        \n"
    "   .      │    x    x            │                 \n"
    "   .      │                      │         o       \n"
    "   .      │  x                   │                 \n"
    "   .      x x         x          │                 \n"
    "   .      │                      │           o     \n"
    "   .    ──┼────────────x────   ──┼─────────────────\n"
    "   .      │                      │                 \n"
    "\n")


def profile8(r):

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.power(1-r**2, -3/2)*np.exp((1.1*r)**2/(r**2 - 1))
    proj = np.sqrt(np.pi)*np.power(1 - r**2, -1/2)*np.exp((1.1*r)**2\
           / (r**2 - 1))/1.1

    return source, proj

profile8.__doc__ = _transform_pairs_docstring.replace(
    "(specific_profile_doc_info)",
    "profile8:\n"
    "Curve B table 2 of Hansen and Law J. Opt. Soc. Am. A 2 510-520 (1985). ::\n\n"
    "   .                   profile8\n"
    "   .          source                projection\n"
    "   .      │                      │                 \n"
    "   .      │                      o oo              \n"
    "   .      │                      │    oo           \n"
    "   .      │                      │       o         \n"
    "   .      │                      │                 \n"
    "   .      x xx xx xx             │        o        \n"
    "   .      │         x            │                 \n"
    "   .      │                      │         o       \n"
    "   .      │           x          │                 \n"
    "   .      │                      │           o     \n"
    "   .    ──┼─────────────────   ──┼─────────────────\n"
    "   .      │                      │                 \n"
    "\n")
