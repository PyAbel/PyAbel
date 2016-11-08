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
from . import dasch
from . import direct
from . import hansenlaw
from . import linbasex
from . import onion_bordas
from . import tools


class Transform(object):
    """Abel transform image class.

    This class provides whole image forward and inverse Abel
    transformations, together with preprocessing (centering, symmetrizing) 
    and post processing (integration) functions. 

    The following class attributes are available, depending on the calculation.
  
    Attributes
    ----------
    transform : numpy 2D array 
        the 2D forward/reverse Abel transform.
    angular_integration : tuple
        (radial-grid, radial-intensity)
        radial coordinates, and the radial intensity (speed) distribution,
        evaluated using :func:`abel.tools.vmi.angular_integration()`.
    residual : numpy 2D array
        residual image (not currently implemented).
    IM: numpy 2D array
        the input image, re-centered (optional) with an odd-size width.
    method : str
        transform method, as specified by the input option.
    direction : str
        transform direction, as specified by the input option.

    Beta : numpy 2D array
        with ``linbasex`` :func:`transform_options=dict(return_Beta=True)`
        Beta array coefficients of Newton sphere spherical harmonics
        
            Beta[0] - the radial intensity variation

            Beta[1] - the anisotropy parameter variation

            ...Beta[n] - higher order terms up to `legedre_orders = [0, ..., n]`
        
    radial : numpy 1d array
        with ``linbasex`` :func:`transform_options=dict(return_Beta=True)`
        radial-grid for Beta array

    projection : 
        with ``linbasex`` :func:`transform_options=dict(return_Beta=True)`
        radial projection profiles at angles `proj_angles`
    """

    _verbose = False

    def __init__(self, IM,
              direction='inverse', method='three_point', center='none',
              symmetry_axis=None, use_quadrants=(True, True, True, True),
              symmetrize_method='average', angular_integration=False,
              transform_options=dict(), center_options=dict(),
              angular_integration_options=dict(),
              recast_as_float64=True, verbose=False):
        """The one stop transform function.

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
                        the recursive algorithm described by Hansen and Law.
            ``basex``
                        the Gaussian "basis set expansion" method
                        of Dribinski et al.
            ``direct``
                        a naive implementation of the analytical
                        formula by Roman Yurchuk.
            ``two_point``
                        the two-point transform of Dasch (1992).
            ``three_point``
                        the three-point transform of Dasch (1992).
            ``onion_bordas``
                        the algorithm of Bordas and co-workers (1996), 
                        re-implemented by Rallis, Wells and co-workers (2014).

            ``onion_peeling``
                        the onion peeling deconvolution as described by 
                        Dasch (1992).

            ``linbasex``
                        the 1d-projections of VM-images in terms of 1d
                        spherical functions by Gerber et al. (2013).

        center : tuple or str
            If a tuple (float, float) is provided, this specifies
            the image center in (y,x) (row, column) format.
            A value `None` can be supplied
            if no centering is desired in one dimension,
            for example 'center=(None, 250)'.
            If a string is provided, an automatic centering algorithm is used

            ``image_center``
                center is assumed to be the center of the image.
            ``convolution``
                center the image by convolution of two projections along each axis.
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

           ``average`` 
                 average the quadrants, in accordance with the `symmetry_axis`
           ``fourier``
                axial symmetry implies that the Fourier components of the 2-D
                projection should be real. Removing the imaginary components
                in reciprocal space leaves a symmetric projection.
                ref: Overstreet, K., et al. 
                "Multiple scattering and the density distribution of a Cs MOT." 
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
            
        angular_integration_options : tuple (or dict)
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
                                    -----o----- (left and right sides equivalent)
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
            analytically. Because the gaussian functions are approximately the 
            size of each pixel, this method also does not make any assumption 
            about the shape of the data. This method is one of the de-facto 
            standards in photoelectron/photoion imaging.

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

        ``linbasex`` *
            VM-images are composed of projected Newton spheres with a common 
            centre. The 2D images are usually evaluated by a decomposition into
            base vectors each representing the 2D projection of a set of 
            particles starting from a centre with a specific velocity 
            distribution. `linbasex` evaluate 1D projections of VM-images in 
            terms of 1D projections of spherical functions, instead.
            
            ..Rev. Sci. Instrum. 84, 033101 (2013): <http://scitation.aip.org/content/aip/journal/rsi/84/3/10.1063/1.4793404>

        ``onion_bordas``
            The onion peeling method, also known as "back projection",
            originates from Bordas *et al.*  `Rev. Sci. Instrum. 67, 2257 (1996)`_.

          .. _Rev. Sci. Instrum. 67, 2257 (1996):  <http://scitation.aip.org/content/aip/journal/rsi/67/6/10.1063/1.1147044> 

            The algorithm was subsequently coded in MatLab by Rallis, Wells and co-workers, `Rev. Sci. Instrum. 85, 113105 (2014)`_.

          .. _Rev. Sci. Instrum. 85, 113105 (2014): <http://scitation.aip.org/content/aip/journal/rsi/85/11/10.1063/1.4899267>

           which was used as the basis of this Python port. See issue `#56`_.

        .. _#56: <https://github.com/PyAbel/PyAbel/issues/56>


        ``onion_peeling`` *
            This is one of the most compact and fast algorithms, with the
            inverse Abel transfrom achieved in one Python code-line, PR #155.
            See also ``three_point`` is the onion peeling algorithm as
            described by Dasch (1992), reference below.

         ``two_point`` *
            Another Dasch method. Simple, and fast, but not as accurate as the
            other methods.

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

        # public class variables
        self.IM = IM   # (optionally) centered, odd-width image
        self.method = method
        self.direction = direction

        # private internal variables
        self._symmetry_axis = symmetry_axis
        self._symmetrize_method = symmetrize_method
        self._use_quadrants = use_quadrants
        self._recast_as_float64 = recast_as_float64
        _verbose = verbose

        # image processing
        self._verify_some_inputs()

        self._center_image(center, **center_options)

        self._abel_transform_image(**transform_options)

        self._integration(angular_integration, transform_options, 
                          **angular_integration_options)

    # end of class instance 

    _verboseprint = print if _verbose else lambda *a, **k: None

    def _verify_some_inputs(self):
        if self.IM.ndim == 1 or np.shape(self.IM)[0] <= 2:
            raise ValueError('Data must be 2-dimensional. \
                              To transform a single row \
                              use the individual transform function.')

        if not np.any(self._use_quadrants):
            raise ValueError('No image quadrants selected to use')

        if not isinstance(self._symmetry_axis, (list, tuple)):
            # if the user supplies an int, make it into a 1-element list:
            self._symmetry_axis = [self._symmetry_axis]

        if self._recast_as_float64:
            self.IM = self.IM.astype('float64')

    def _center_image(self, center, **center_options):
        if center != "none":
            self.IM = tools.center.center_image(self.IM, center, 
                                                 **center_options)

    def _abel_transform_image(self, **transform_options):
        if self.method == "linbasex" and self._symmetry_axis is not None: 
            self._abel_transform_image_full(**transform_options)
        else:
            self._abel_transform_image_by_quadrant(**transform_options)


    def _abel_transform_image_full(self, **transform_options):

        abel_transform = {
            # "basex": basex.basex_transform,
            "linbasex": linbasex.linbasex_transform_full
        }
        t0 = time.time()

        self._verboseprint('Calculating {0} Abel transform using {1} method -'
                          .format(self.direction, self.method), 
                          '\n    image size: {:d}x{:d}'.format(*self.IM.shape))

        self.transform, radial, Beta, QLz = abel_transform[self.method](self.IM,
                                                   **transform_options)

        self._verboseprint("{:.2f} seconds".format(time.time()-t0))

        self.Beta = Beta
        self.projection = QLz
        self.radial = radial
            

    def _abel_transform_image_by_quadrant(self, **transform_options):

        abel_transform = {
            "basex": basex.basex_transform,
            "direct": direct.direct_transform,
            "hansenlaw": hansenlaw.hansenlaw_transform,
            "linbasex": linbasex.linbasex_transform,
            "onion_bordas": onion_bordas.onion_bordas_transform,
            "onion_peeling": dasch.onion_peeling_transform,
            "two_point": dasch.two_point_transform,
            "three_point": dasch.three_point_transform,
        }

        self._verboseprint('Calculating {0} Abel transform using {1} method -'
                          .format(self.direction, self.method), 
                          '\n    image size: {:d}x{:d}'.format(*self.IM.shape))

        t0 = time.time()

        # split image into quadrants
        Q0, Q1, Q2, Q3 = tools.symmetry.get_image_quadrants(
                         self.IM, reorient=True,
                         symmetry_axis=self._symmetry_axis, 
                         symmetrize_method=self._symmetrize_method)

        def selected_transform(Z):
            return abel_transform[self.method](Z, direction=self.direction, 
                                               **transform_options)

        AQ0 = AQ1 = AQ2 = AQ3 = None

        # Inverse Abel transform for quadrant 1 (all include Q1)
        AQ1 = selected_transform(Q1)

        if 0 in self._symmetry_axis:
            AQ2 = selected_transform(Q2)

        if 1 in self._symmetry_axis:
            AQ0 = selected_transform(Q0)

        if None in self._symmetry_axis:
            AQ0 = selected_transform(Q0)
            AQ2 = selected_transform(Q2)
            AQ3 = selected_transform(Q3)

        if self.method == "linbasex" and\
           "return_Beta" in transform_options.keys():
            # linbasex evaluates speed and anisotropy parameters
            # AQi == AIM, R, Beta, QLz
            Beta0 = AQ0[2]
            Beta1 = AQ1[2]
            Beta2 = AQ2[2]
            Beta3 = AQ3[2]
            # rconstructed images of each quadrant
            AQ0 = AQ0[0]
            AQ1 = AQ1[0]
            AQ2 = AQ2[0]
            AQ3 = AQ3[0]
            # speed
            self.linbasex_angular_integration = self.Beta[0]\
                 (Beta0[0] + Beta1[0] + Beta2[0] + Beta3[0])/4
            # anisotropy
            self.linbasex_anisotropy_parameter = self.Beta[1]\
                 (Beta0[1] + Beta1[1] + Beta2[1] + Beta3[1])/4

        # reassemble image
        self.transform = tools.symmetry.put_image_quadrants(
                                (AQ0, AQ1, AQ2, AQ3), 
                                original_image_shape=self.IM.shape,
                                symmetry_axis=self._symmetry_axis)

        self._verboseprint("{:.2f} seconds".format(time.time()-t0))

    def _integration(self, angular_integration, transform_options,
                     **angular_integration_options):
        if angular_integration:
            if 'dr' in transform_options and\
               'dr' not in angular_integration_options:
                # assume user forgot to pass grid size
                angular_integration_options['dr'] = transform_options['dr']

            self.angular_integration = tools.vmi.angular_integration(
                                             self.transform,
                                             **angular_integration_options)
