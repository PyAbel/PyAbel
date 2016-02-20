# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import time
import warnings


def transform(
    IM, direction='inverse', method='three_point', center='none',
        verbose=True, vertical_symmetry=True,
        horizontal_symmetry=True, use_quadrants=(True, True, True, True),
        transform_options={}, center_options={}):
    """
    transform() is the go-to function for all of your Abel transform needs!!

    This performs the forward or reverse Abel transform
    using a user-selected method.


    Parameters
    ----------
    IM : a NxM numpy array
        This is the image to be transformed
    direction : str
        The type of Abel transform to be performed.

        'forward'
                    A 'forward' Abel transform takes a (2D) slice of a 3D image
                    and returns the 2D projection.
        'inverse'
                    An 'inverse' Abel transform takes a 2D projection
                    and reconstructs a 2D slice of the 3D image.

        The default is 'inverse'.
    method : str
        specifies which numerical approximation to the Abel transform
        should be employed (see below). The options are

        'hansenlaw'
                    the recursive algorithm described by Hansen and Law
        'basex'
                    the Gaussian "basis set expansion" method
                    of Dribinski et al.
        'direct'
                    a naive implementation of the analytical
                    formula by Roman Yurchuk.
        'three_point'
                    the three-point transform of Dasch and co-workers
    center : tuple or str
        If a tuple (float, float) is provided, this specifies
        the image center in (y,x) (row, column) format.
        A value `None` can be supplied
        if no centering is desired in one dimension,
        for example 'center=(None, 250)'.
        If a string is provided, an automatic centering algorithm is used

        'image_center'
                    center is assumed to be the center of the image.
        'by_slice'
                    (whatever this does)
        'com'
                    the center is calculated as the center of mass
        'none'
                     (Default)
                     No centering is performed. An image with an odd
                     number of columns must be provided.
    verbose : boolean
        True/False to determine if non-critical output should be printed.
    vertical_symmetry : boolean
        Symmetrize the image in the up/down direction
        (The first axis is the vertical axis.)
    horizontal_symmetry : boolean
        Symmetrize the image in the left/right direction?
    use_quadrants : boolean tuple (Q0,Q1,Q2,Q3)
        select quadrants to be used in the analysis.
        The quadrants are numbered starting from
        Q0 in the upper right and proceeding counter-clockwise:

        ::

             +--------+--------+
             | Q1   * | *   Q0 |
             |   *    |    *   |
             |  *     |     *  |                               AQ1 | AQ0
             +--------o--------+ --(inverse Abel transform)--> ----o----
             |  *     |     *  |                               AQ2 | AQ3
             |   *    |    *   |
             | Q2  *  | *   Q3 |          AQi == inverse Abel transform
             +--------+--------+                 of quadrant Qi


        (1) vertical_symmetry = True

        ::

           Combine:  `Q01 = Q1 + Q2, Q23 = Q2 + Q3`
           inverse image   AQ01 | AQ01
                           -----o-----
                           AQ23 | AQ23

        (2) horizontal_symmetry = True

        ::

           Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
           inverse image   AQ12 | AQ03
                           -----o-----
                           AQ12 | AQ03

        (3) vertical_symmetry = True, horizontal = True

        ::

           Combine: Q = Q0 + Q1 + Q2 + Q3
           inverse image   AQ | AQ
                           ---o---  all quadrants equivalent
                           AQ | AQ
       ::

    transform_options : tuple
        Additional arguments passed to the individual transform functions.
        See the documentation for the individual transform method for options.
    center_options : tuple
        Additional arguments to be passed to the centering function.

    Transform Methods
    -----------------
    As mentioned above, PyAbel offers several different approximations
    to the the exact abel transform.
    All the the methods should produce similar results, but
    depending on the level and type of noise found in the image,
    certain methods may perform better than others.

    'hansenlaw' - This "recursive algorithm" produces reliable results
        and is quite fast (~0.1 sec for a 1001x1001 image).
        It makes no assumptions about the data
        (apart from cylindrical symmetry). It tends to require that the data
        is finely sampled for good convergence.

        E. W. Hansen and P.-L. Law "Recursive methods for computing
        the Abel transform and its inverse"
        J. Opt. Soc. A*2, 510-520 (1985)
        http://dx.doi.org/10.1364/JOSAA.2.000510

    'basex'* - The "basis set exapansion" algorithm describes the data in terms
        of gaussian functions, which themselves can be abel transformed
        analytically. Because the gaussian functions are approximately the size
        of each pixel, this method also does not make any assumption about
        the shape of the data. This method is one of the de-facto standards in
        photoelectron/photoion imaging.

         Dribinski et al, 2002 (Rev. Sci. Instrum. 73, 2634)
         http://dx.doi.org/10.1063/1.1482156

    'direct' - This method attempts a direct integration of the Abel
        transform integral. It makes no assumptions about the data
        (apart from cylindrical symmetry),
        but it typically requires fine sampling to converge.
        Such methods are typically inefficient,
        but thanks to this Cython implementation (by Roman Yurchuk),
        this 'direct' method is competitive with the other methods.

    'three_point'* - The "Three Point" Abel transform method
        exploits the observation that the value of the Abel inverted data
        at any radial position r is primarily determined from changes
        in the projection data in the neighborhood of r.
        This method is also very efficient
        once it has generated the basis sets.

        Dasch, 1992 (Applied Optics, Vol 31, No 8, March 1992, Pg 1146-1152).

    *   The methods marked with a * indicate methods that generate basis sets.
        The first time they are run for a new image size,
        it takes seconds to minutes to generate the basis set.
        However, this basis set is saved to disk can can be reloaded,
        meaning that future transforms are performed
        much more quickly.


    Returns
    -------
    results : dict
        The transform function returns a dictionary of results
        depending on the options selected

        'results['transform']'
                (always returned) is the 2D forward/reverse Abel transform
        'results['radial_intensity']'
                is not currently implemented
        'results['residual']'
                is not currently implemented
    """

    verboseprint = print if verbose else lambda *a, **k: None

    if IM.ndim == 1 or np.shape(IM)[0] <= 2:
            raise ValueError('Data must be 2-dimensional. \
                              To transform a single row \
                              use the individual transform function.')

    if not np.any(use_quadrants):
        raise ValueError('No image quadrants selected to use')
    rows, cols = np.shape(IM)

    # centering:
    if center == 'none':  # no centering
        if rows % 2 != 1:
            raise ValueError('Image must have an odd number of columns. \
                              Use a centering method.')
    else:
        IM = abel.center_image(IM, center)

    #########################

    verboseprint('Calculating {0} Abel transform using {1} method -'.format(
      direction, method), 'image size: {:d}x{:d}'.format(rows, cols))

    t0 = time.time()

    # split image into quadrants
    Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(
      IM, reorient=True, vertical_symmetry=vertical_symmetry,
      horizontal_symmetry=horizontal_symmetry)

    def selected_transform(Z):
        if method == 'hansenlaw':
            if direction == 'forward':
                return abel.hansenlaw.fabel_hansenlaw(Z, **transform_options)
            elif direction == 'inverse':
                return abel.hansenlaw.iabel_hansenlaw(Z, **transform_options)

        elif method == 'three_point':
            if direction == 'forward':
                raise ValueError('Forward three-point not implemented')
            elif direction == 'inverse':
                return abel.three_point.iabel_three_point_transform(
                  Z, **transform_options)

        elif method == 'basex':
            if direction == 'forward':
                raise ValueError('Forward basex not implemented')
            elif direction == 'inverse':
                return abel.basex.iabel_basex(Z, **transform_options)

        elif method == 'direct':
            if direction == 'forward':
                raise ValueError('Coming soon...')
            elif direction == 'inverse':
                raise ValueError('Coming soon...')

    AQ0 = AQ1 = AQ2 = AQ3 = None
    # Inverse Abel transform for quadrant 1 (all include Q1)
    AQ1 = selected_transform(Q1)

    if vertical_symmetry:
        AQ2 = selected_transform(Q2)

    if horizontal_symmetry:
        AQ0 = selected_transform(Q0)

    if not vertical_symmetry and not horizontal_symmetry:
        AQ0 = selected_transform(Q0)
        AQ2 = selected_transform(Q2)
        AQ3 = selected_transform(Q3)

    # reassemble image
    results = {}
    results['transform'] = abel.tools.symmetry.put_image_quadrants(
      (AQ0, AQ1, AQ2, AQ3), odd_size=cols % 2,
      vertical_symmetry=vertical_symmetry,
      horizontal_symmetry=horizontal_symmetry)

    verboseprint("{:.2f} seconds".format(time.time()-t0))

    return results


def main():
    import matplotlib.pyplot as plt
    IM0 = abel.tools.analytical.sample_image_dribinski(n=201)
    IM1 = transform(IM0, direction='forward', center='com',
                    method='hansenlaw')['transform']
    IM2 = transform(IM1, direction='inverse', method='basex')['transform']

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))

    axs[0, 0].imshow(IM0)
    axs[0, 1].imshow(IM1)
    axs[0, 2].imshow(IM2)

    axs[1, 0].plot(*abel.tools.vmi.angular_integration(IM0))
    axs[1, 1].plot(*abel.tools.vmi.angular_integration(IM1))
    axs[1, 2].plot(*abel.tools.vmi.angular_integration(IM2))

    plt.show()

if __name__ == "__main__":
    main()
