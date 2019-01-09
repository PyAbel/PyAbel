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

__doc__ = """
Analytical function Abel-transform pairs

profiles 1--7, table 1 of:
    `G. C.-Y Chan and G. M. Hieftje Spectrochimica Acta B 61, 31–41 (2006)
    <https://doi.org/10.1016/j.sab.2005.11.009>`_

Note:
    the transform pair functions are more conveniently accessed through
    :class:`abel.tools.analytical.TransformPair`::

        func = abel.tools.analytical.TransformPair(n, profile=nprofile)

    which sets the radial range `r` and provides attributes
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
    `Cremers and Birkebak App. Opt. 5, 1057–1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

    .. math::

        \epsilon(r) &= 0.75 + 12r^2 - 32r^3  & 0 \le r \le 0.25

        \epsilon(r) &= \\frac{16}{27}(1 + 6r - 15r^2 + 8r^3)
                    & 0.25 \lt r \le 1

        I(r) &= \\frac{1}{108}(128a_1 +a_{0.25}) + \\frac{2}{27}r^2
                  (283a_{0.25} - 112a_1) +

        & \,\,\,\, \\frac{8}{9}r^2\left[4(1+r^2)\ln\\frac{1+a_1}{r} -
          (4+31r^2)\ln\\frac{0.25+a_{0.25}}{r}\\right] &  0 \le r \le 0.25

        I(r) &= \\frac{32}{27}\left[a_1 - 7a_1 r + 3r^2(1+r^2)
                \ln\\frac{1+a_1}{r}\\right]  & 0.25 \lt r \le 1

    ..
              source                projection
        ┼+1.3                  ┼+1.3               
        │                      o   o               
        │     x                │     o             
        │   x   x              │       o           
        │ x                    │                   
        x         x            │         o         
        │           x          │                   
        │                      │           o       
        │             x        │                   
        │                      │             o     
        ┼+0─────────────x──┼   ┼+0─────────────o──┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(1)
    """

    if np.any(r <= 0) or np.any(r > 1):
        raise ValueError('r must be 0 < r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # left side r <= 0.25
    rl = r[r <= 0.25]

    # source
    source_l = 3/4 + 12*rl**2 - 32*rl**3

    # projection
    a4l = a(0.25, rl)
    a1l = a(1, rl)
    rl2 = rl**2

    proj_l = (128*a1l + a4l)/108 + (283*a4l - 112*a1l)*rl2*2/27 +\
             (4*(1 + rl2)*np.log((1 + a1l)/rl) -
             (4 + 31*rl2)*np.log((0.25 + a4l)/rl))*rl2*8/9

    # right side r > 0.25
    rr = r[r > 0.25]

    # source
    source_r = (16/27)*(1 + 6*rr - 15*rr**2 + 8*rr**3)

    # projection
    a1r = a(1, rr)
    rr2 = rr**2

    proj_r = (a1r - 7*a1r*rr2 + 3*rr2*(1 + rr2)*np.log((1 + a1r)/rr))*32/27

    source = np.concatenate((source_l, source_r))
    projection = np.concatenate((proj_l, proj_r))

    return source, projection


def profile2(r):
    """**profile2**:
    `Cremers and Birkebak App. Opt. 5, 1057–1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

    .. math::

        \epsilon(r) &= 1 - 3r^2 + 2r^3 & 0 \le r \le 1

        I(r) &= a_1\left(1-\\frac{5}{2}r^2\\right) +
                \\frac{3}{2}r^4\ln\\frac{1+a_1}{r} & 0 \le r \le 1

    ..
              source                projection
        ┼+1.1                  ┼+1.1               
        x x                    o o                 
        │   x                  │   o               
        │     x                │     o             
        │       x              │                   
        │                      │       o           
        │         x            │                   
        │                      │         o         
        │           x          │           o       
        │             x        │                   
        ┼+0─────────────x──┼   ┼+0───────────o────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(2)
    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = 1 - 3*r*r + 2*r**3

    a1 = a(1, r)
    projection = a1*(1 - r**2*5/2) + r**4*np.log((1 + a1)/r)*3/2

    return source, projection


def profile3(r):
    """**profile3**:
    `Cremers and Birkebak App. Opt. 5, 1057–1064 (1966) Eq(13)
    <https://doi.org/10.1364/AO.5.001057>`_

    .. math::

        \epsilon(r) &= 1-2r^2  & 0 \le r \le 0.5

        \epsilon(r) &= 2(1-r^2)^2 & 0.5 \lt r \le 1

        I(r) &= \\frac{4a_1}{3}(1+2r^2)-\\frac{2 a_{0.5}}{3}(1+8r^2) -
                4r^2\ln\\frac{1-a_1}{0.5+a_{0.5}} & 0 \le r \le 0.5

        I(r) &= \\frac{4a_1}{3}(1+2r^2)-4r^2\ln\\frac{1-a_1}{r} &
                0.5 \lt r \le 1

    ..
              source                projection
        ┼+1.1                  ┼+1.1               
        x x x                  o o                 
        │                      │   o               
        │     x                │     o             
        │       x              │                   
        │                      │       o           
        │         x            │                   
        │                      │         o         
        │           x          │                   
        │                      │           o       
        ┼+0───────────x────┼   ┼+0───────────o────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(3)
    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # left side r <= 0.5
    rl = r[r <= 0.5]

    source_l = 1 - 2*rl**2

    a5l = a(0.5, rl)
    a1l = a(1, rl)
    # power rm**2 typo in Cremers
    proj_l = (4/3)*a1l*(1 + 2*rl**2) - (2/3)*a5l*(1 + 8*rl**2) -\
             4*rl**2*np.log((1 + a1l)/(0.5 + a5l))

    # right side r > 0.5
    rr = r[r > 0.5]
    a1r = a(1, rr)

    source_r = 2*(1 - rr)**2
    proj_r = (4/3)*a1r*(1 + 2*rr**2) - 4*rr**2*np.log((1 + a1r)/rr)

    source = np.concatenate((source_l, source_r))
    projection = np.concatenate((proj_l, proj_r))

    return source, projection


def profile4(r):
    """**profile4**: `Alvarez, Rodero, Quintero Spectochim. Acta B 57,
    1665–1680 (2002) <https://doi.org/10.1016/S0584-8547(02)00087-3>`_

    Note:
        Published projection has misprints
        (“19\ **3**\ .30083” instead of “19\ **6**\ .30083” in both cases).

    .. math::

        \epsilon(r) &= 0.1 + 5.51r^2 - 5.25r^3 & 0 \le r \le 0.7

        \epsilon(r) &= -40.74 + 155.56r - 188.89r^2 + 74.07r^3
                    & 0.7 \lt r \le1

        I(r) &= 22.68862a_{0.7} - 14.811667a_1 + (217.557a_{0.7} -
        196.30083a_1)r^2 +

          & \,\,\, 155.56r^2\ln\\frac{1 + a_1}{0.7 + a_{0.7}} +
            r^4\left(55.5525\ln\\frac{1 + a_1}{r} - 59.49\ln\\frac{0.7 +
            a_{0.7}}{r}\\right)  & 0 \le r \le 0.7

        I(r) &= -14.811667a_1 - 196.30083a_1 r^2 + r^2(155.56 + 55.5525r^2)
                \ln\\frac{1 + a_1}{r} & 0.7 \lt r \le 1

    ..
              source                projection
        ┼+2.2                  ┼+2.2       o       
        │                      │         o   o     
        │                      │       o       o   
        │                      │     o             
        │                      │                   
        │                      │ o o             o 
        │           x x        o                   
        │         x     x      │                   
        │       x              │                   
        │     x                │                   
        ┼+0─x─────────────x┼   ┼+0────────────────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(4)
    """

    def source_left(x):
        """Profile4 source x <= 0.7.

        """

        return 0.1 + 5.51*x**2 - 5.25*x**3

    def source_right(x):
        """Profile4 source x > 0.7.

        """

        return -40.74 + 155.56*x - 188.89*x**2 + 74.07*x**3

    def proj_left(x):
        """Profile4 projection x < 0.7 of right function part.

        """

        a7 = a(0.7, x)
        a1 = a(1, x)
        return 22.68862*a7 - 14.811667*a1 + (217.557*a7 - 196.30083*a1)*x**2 +\
               +155.56*x**2*np.log((1 + a1)/(0.7 + a7)) +\
               x**4*(55.5525*np.log((1 + a1)/x) - 59.49*np.log((0.7 + a7)/x))

    def proj_right(x):
        """Profile4 projection x > 0.7.

        """

        a1 = a(1, x)
        return -14.811667*a1 - 196.30083*a1*x**2 +\
               x**2*(155.56 + 55.5525*x**2)*np.log((1 + a1)/x)

    if np.any(r <= 0) or np.any(r > 1):
        raise ValueError('r must be 0 < r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    # left side r <= 0.7 of source, projection profile
    rl = r[r <= 0.7]

    source_l = source_left(rl)
    proj_l = proj_left(rl)

    # right side r > 0.7 of source, projection profile
    rr = r[r > 0.7]

    source_r = source_right(rr)
    proj_r = proj_right(rr)

    source = np.concatenate((source_l, source_r))
    projection = np.concatenate((proj_l, proj_r))

    return source, projection


def profile5(r):
    """**profile5**: `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55,
    231–243 (1996) <https://doi.org/10.1016/j.amc.2014.03.043>`_

    .. math::

        \epsilon(r) &= 1 & 0 \le r \le 1

        I(r) &= 2a_1 & 0 \le r \le 1

    ..
              source                projection
        ┼+2.1                  ┼+2.1               
        │                      │     o o           
        │                      │         o o       
        │                      │             o     
        │                      │                   
        │                      │               o   
        x x x x x x x x x x    │                   
        │                      │                 o 
        │                      │                   
        │                      │                   
        ┼+0────────────────┼   ┼+0────────────────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(5)
    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.ones_like(r)
    projection = 2*a(1, r)

    return source, projection


def profile6(r):
    """**profile6**: `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55,
    231–243 (1996) <https://doi.org/10.1016/j.amc.2014.03.043>`_

    .. math::

        \epsilon(r) &= (1-r^2)^{-\\frac{3}{2}} \exp\left[1.1^2\left(
                        1 - \\frac{1}{1-r^2}\\right)\\right] & 0 \le r \le 1

        I(r) &= \\frac{\sqrt{\pi}}{1.1a_1} \exp\left[1.1^2\left(
                        1 - \\frac{1}{1-r^2}\\right)\\right] & 0 \le r \le 1

    ..
              source                projection
        ┼+1.8                  ┼+1.8               
        │                      o o o               
        │                      │     o o           
        │                      │         o         
        │                      │                   
        x x x x x x x          │           o       
        │             x        │                   
        │                      │             o     
        │               x      │                   
        │                      │               o   
        ┼+0────────────────┼   ┼+0────────────────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(6)
    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = np.exp(1.1**2*(1 - 1/(1 - r**2)))/np.sqrt(1 - r**2)**3
    projection = np.exp(1.1**2*(1 - 1/(1 - r**2)))*np.sqrt(np.pi)/1.1/a(1, r)

    return source, projection


def profile7(r):
    """**profile7**:
    `Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231–243 (1996)
    <https://doi.org/10.1016/j.amc.2014.03.043>`_

    .. math::

        \epsilon(r) &= \\frac{1}{2}(1+10r^2-23r^4+12r^6) & 0 \le r \le 1

        I(r) &= \\frac{8}{105}a_1(19 + 34r^2 - 125r^4 + 72r^6) & 0 \le r \le 1

    ..
              source                projection
        ┼+1.7                  ┼+1.7               
        │                      o o o o o           
        │                      │         o         
        │                      │                   
        │       x x x          │           o       
        │     x       x        │                   
        │                      │             o     
        │   x                  │                   
        x x             x      │                   
        │                      │               o   
        ┼+0───────────────x┼   ┼+0────────────────┼
        0          r      +1   0          r      +1

    .. plot::

        from tools.transform_pairs import plot
        plot(7)
    """

    if np.any(r < 0) or np.any(r > 1):
        raise ValueError('r must be 0 <= r <= 1')

    if not hasattr(r, '__len__'):
        r = np.asarray([r])

    source = (1 + 10*r**2 - 23*r**4 + 12*r**6)/2
    projection = a(1, r)*(19 + 34*r**2 - 125*r**4 + 72*r**6)*8/105

    return source, projection
