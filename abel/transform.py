#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import time
import warnings

from . import basex
from . import hansenlaw
from . import direct
from . import three_point
from . import tools


class transform(object):
    """Abel transform image class

    This class provides whole image forward and inverse Abel
    transformations, together with preprocessing (centering, symmetrizing) 
    and post processing (integration) functions.
  
    """

    def __init__(self, IM,
              direction='inverse', method='three_point', center='none',
              symmetry_axis=None, use_quadrants=(True, True, True, True),
              symmetrize_method='average', angular_integration=False,
              transform_options=dict(), center_options=dict(),
              angular_integration_options=dict(),
              recast_as_float64=True, verbose=False):

        """
        Parameters
        ----------
        IM : a NxM numpy array
            This is the image to be transformed

        direction : str
            The type of Abel transform to be performed.

            ``forward``
                A 'forward' Abel transform takes a (2D) slice of a 3D image
                and returns the 2D projection.

            ``inverse``
                An 'inverse' Abel transform takes a 2D projection
                and reconstructs a 2D slice of the 3D image.

            The default is ``inverse``.

        method : str
            specifies which numerical approximation to the Abel transform
            should be employed (see below). The options are

            ``hansenlaw``
                        the recursive algorithm described by Hansen and Law
            ``basex``
                        the Gaussian "basis set expansion" method
                        of Dribinski et al.
            ``direct``
                        a naive implementation of the analytical
                        formula by Roman Yurchuk.
            ``three_point``
                        the three-point transform of Dasch and co-workers

        center : tuple or str
            If a tuple (float, float) is provided, this specifies
            the image center in (y,x) (row, column) format.
            A value `None` can be supplied
            if no centering is desired in one dimension,
            for example 'center=(None, 250)'.
            If a string is provided, an automatic centering algorithm is used

            ``image_center``
                center is assumed to be the center of the image.
            ``slice``
                the center is found my comparing slices in the horizontal and
                vertical directions
            ``com``
                the center is calculated as the center of mass
            ``gaussian``
                the center is found using a fit to a Gaussian function. This
                only makes sense if your data looks like a Gaussian.
            ``none``
                (Default)
                No centering is performed. An image with an odd
                number of columns must be provided.

        symmetry_axis : None, int or tuple
            Symmetrize the image about the numpy axis 
            0 (vertical), 1 (horizontal), (0,1) (both axes)
            
        use_quadrants : tuple of 4 booleans
            select quadrants to be used in the analysis: (Q0,Q1,Q2,Q3).
            Quadrants are numbered counter-clockwide from upper right.
            See note below for description of quadrants. 
            Default is ``(True, True, True, True)``, which uses all quadrants.

        symmetrize_method: str
           Method used for symmetrizing the image.

           average: Simply average the quadrants.
           fourier: Axial symmetry implies that the Fourier components of the 2-D
                    projection should be real. Removing the imaginary components in
                    reciprocal space leaves a symmetric projection.
                    ref: Overstreet, K., et al. "Multiple scattering and the density
                         distribution of a Cs MOT." 
                         Optics express 13.24 (2005): 9672-9682.
                         http://dx.doi.org/10.1364/OPEX.13.009672

        angular_integration: boolean
            integrate the image over angle to give the radial (speed) intensity
            distribution

        transform_options : tuple
            Additional arguments passed to the individual transform functions.
            See the documentation for the individual transform method for options.

        center_options : tuple
            Additional arguments to be passed to the centering function.
            
            
        .. note:: Quadrant combining 
             The quadrants can be combined (averaged) using the 
             ``use_quadrants`` keyword in order to provide better data quality.
             
             The quadrants are numbered starting from
             Q0 in the upper right and proceeding counter-clockwise: ::

                  +--------+--------+
                  | Q1   * | *   Q0 |
                  |   *    |    *   |
                  |  *     |     *  |                               AQ1 | AQ0
                  +--------o--------+ --(inverse Abel transform)--> ----o----
                  |  *     |     *  |                               AQ2 | AQ3
                  |   *    |    *   |
                  | Q2  *  | *   Q3 |          AQi == inverse Abel transform
                  +--------+--------+                 of quadrant Qi
                  
        
            Three cases are possible: 
            
            1) symmetry_axis = 0 (vertical): ::

                    Combine:  Q01 = Q0 + Q1, Q23 = Q2 + Q3
                    inverse image   AQ01 | AQ01
                                    -----o-----
                                    AQ23 | AQ23


            2) symmetry_axis = 1 (horizontal): ::

                    Combine: Q12 = Q1 + Q2, Q03 = Q0 + Q3
                    inverse image   AQ12 | AQ03
                                    -----o----- (top and bottom equivalent)
                                    AQ12 | AQ03
                            

            3) symmetry_axis = (0, 1) (both): ::

                    Combine: Q = Q0 + Q1 + Q2 + Q3
                    inverse image   AQ | AQ
                                    ---o---  (all quadrants equivalent)
                                    AQ | AQ

        angular_integration_options: tuple (or dict)
            Additional arguments passed to the angular_integration transform 
            functions.  See the documentation for angular_integration for options.

        recast_as_float64 : boolean
            True/False that determines if the input image should be recast to 
            ``float64``. Many images are imported in other formats (such as 
            ``uint8`` or ``uint16``) and this does not always play well with the 
            transorm algorithms. This should probably always be set to True. 
            (Default is True.)

        verbose : boolean
            True/False to determine if non-critical output should be printed.

        Returns
        -------
        IMobj : class elements
            The transform function returns class element results
            depending on the options selected

            ``IMobj.transform``
                    the 2D forward/reverse Abel transform
            ``IMobj.angular_integration``
                    tuple: radial coordinates, 
                           radial intensity (speed) distribution
            ``IMobj.residual``
                    residual image is not currently implemented
            ``IMobj.method``
                    transform method
            ``IMobj.direction``
                    transform direction
        
        Notes
        -----
        As mentioned above, PyAbel offers several different approximations
        to the the exact abel transform.
        All the the methods should produce similar results, but
        depending on the level and type of noise found in the image,
        certain methods may perform better than others. Please see the 
        "Transform Methods" section of the documentation for complete information.

        ``hansenlaw`` 
            This "recursive algorithm" produces reliable results
            and is quite fast (~0.1 sec for a 1001x1001 image).
            It makes no assumptions about the data
            (apart from cylindrical symmetry). It tends to require that the data
            is finely sampled for good convergence.

            E. W. Hansen and P.-L. Law "Recursive methods for computing
            the Abel transform and its inverse"
            J. Opt. Soc. A*2, 510-520 (1985)
            http://dx.doi.org/10.1364/JOSAA.2.000510

        ``basex`` * 
            The "basis set exapansion" algorithm describes the data in terms
            of gaussian functions, which themselves can be abel transformed
            analytically. Because the gaussian functions are approximately the size
            of each pixel, this method also does not make any assumption about
            the shape of the data. This method is one of the de-facto standards in
            photoelectron/photoion imaging.

             Dribinski et al, 2002 (Rev. Sci. Instrum. 73, 2634)
             http://dx.doi.org/10.1063/1.1482156

        ``direct``
            This method attempts a direct integration of the Abel
            transform integral. It makes no assumptions about the data
            (apart from cylindrical symmetry),
            but it typically requires fine sampling to converge.
            Such methods are typically inefficient,
            but thanks to this Cython implementation (by Roman Yurchuk),
            this 'direct' method is competitive with the other methods.

        ``three_point`` *
            The "Three Point" Abel transform method
            exploits the observation that the value of the Abel inverted data
            at any radial position r is primarily determined from changes
            in the projection data in the neighborhood of r.
            This method is also very efficient
            once it has generated the basis sets.

            Dasch, 1992 (Applied Optics, Vol 31, No 8, March 1992, Pg 1146-1152).

        ``*``
            The methods marked with a * indicate methods that generate basis sets.
            The first time they are run for a new image size,
            it takes seconds to minutes to generate the basis set.
            However, this basis set is saved to disk can can be reloaded,
            meaning that future transforms are performed
            much more quickly.

        """

        self.IM = IM
        self.method = method
        self.direction = direction
 
        abel_transform = {
            "basex" : basex.basex_transform,
            "direct" : direct.direct_transform,
            "hansenlaw" : hansenlaw.hansenlaw_transform,
            "three_point" : three_point.three_point_transform,
        }

        verboseprint = print if verbose else lambda *a, **k: None

        if IM.ndim == 1 or np.shape(IM)[0] <= 2:
            raise ValueError('Data must be 2-dimensional. \
                              To transform a single row \
                              use the individual transform function.')

        if not np.any(use_quadrants):
            raise ValueError('No image quadrants selected to use')

        rows, cols = np.shape(IM)

        if not isinstance(symmetry_axis, (list, tuple)):
            # if the user supplies an int, make it into a 1-element list:
            symmetry_axis = [symmetry_axis]

        if recast_as_float64:
            IM = IM.astype('float64')

        # centering:
        if center == 'none':  # no centering
            if cols % 2 != 1:
                raise ValueError('Image must have an odd number of columns. \
                              Use a centering method.')
        else:
            IM = tools.center.center_image(IM, center, **center_options)

        #########################

        verboseprint('Calculating {0} Abel transform using {1} method -'
                     .format(direction, method), 
                     '\n    image size: {:d}x{:d}'.format(rows, cols))

        t0 = time.time()

        # split image into quadrants
        Q0, Q1, Q2, Q3 = tools.symmetry.get_image_quadrants(
                         IM, reorient=True, symmetry_axis=symmetry_axis, 
                         symmetrize_method=symmetrize_method)

        def selected_transform(Z):
            return abel_transform[method](Z, direction=direction, 
                                          **transform_options)

        AQ0 = AQ1 = AQ2 = AQ3 = None

        # Inverse Abel transform for quadrant 1 (all include Q1)
        AQ1 = selected_transform(Q1)
    
        if 0 in symmetry_axis:
            AQ2 = selected_transform(Q2)

        if 1 in symmetry_axis:
            AQ0 = selected_transform(Q0)

        if None in symmetry_axis:
            AQ0 = selected_transform(Q0)
            AQ2 = selected_transform(Q2)
            AQ3 = selected_transform(Q3)

        # reassemble image
        self.transform = tools.symmetry.put_image_quadrants((AQ0, AQ1, AQ2, AQ3), 
                                original_image_shape=IM.shape,
                                symmetry_axis=symmetry_axis)

        verboseprint("{:.2f} seconds".format(time.time()-t0))

        #########################

        # radial intensity distribution
        if angular_integration:
            if 'dr' in transform_options and\
               'dr' not in angular_integration_options:
                # assume user forgot to pass grid size
                angular_integration_options['dr'] = transform_options['dr']

            self.angular_integration = tools.vmi.angular_integration(
                             self.transform, **angular_integration_options)
