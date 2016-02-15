# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import time
import warnings

class AbelTransform(object):
    """AbelTransform class
    
    This class provides a conveinent way to call all of transform functions
    as well as the preprocessing (centering) and post processing 
    (integration) functions. 
    
    
    Attributes
        ----------
        IM : NxM numpy array
            the original 2D image supplied
        transform : NxM numpy array
            the transformed (either forward or inverse) image

    """
    def __init__ (self, IM, direction=None ,method='three_point', center='none', verbose=True,
                 vertical_symmetry=True, horizontal_symmetry=True, integrate=True, use_quadrants=(True,True,True,True),
                 transform_options=(), center_options=() ):
        """__init__ initializing the AbelTransform class performs a forward (or reverse) abel transform

        This performs the forward or reverse Abel transform using a user-selected method.
        AbelTransform is typically not called directly, usually the 'abel' or 'iabel' functions are called,
        which specify the forward or reverse abel transform.
        
        Transform Methods
        --------
        PyAbel offers several different approximations to the the exact abel transform 
        ...<provide references here>
       

        Parameters
        ----------
        IM : a NxM numpy array
            This is the image to be transformed
        direction : 'forward' or 'inverse'
            The type of Abel transform to be performed. 
            A 'forward' Abel transform takes a (2D) slice of a 3D image and returns the 2D projection.
            An 'inverse' Abel transform takes a 2D projection and reconstructs a 2D slice of the 3D image.
        method : str
            The method specifies which numerical approximation to the Abel transform should be employed.
            All the the methods should produce similar results, but depending on the level and type of noise
            found in the image, certain methods may perform better than others.
            The options are:
                'hansenlaw' - the recursive algorithm described by Hansen and Law
                'basex' - the Gaussian "basis set expansion" method of Dribinski et al.
                'direct' - a naive implementation of the analytical formula by Roman Yurchuk. 
                'three_point' - the three-point transform of Dasch and co-workers
        center : tuple or str
            If a tuple (float, float) is provides, this specifies the image center in (y,x) 
            (row, column) format. A value None can be supplied if no centering is desired in one dimention,
            for example center=(None, 250).
            If a string is provided, an automatic centering algorithm is used:
                'image_center' - the center is assumed to be the center of the image. 
                'by_slice' 
                'com' - the center is calculated as the center of mass
                'none' - no centering is performed. An image with an odd number of columns must be provided.
                (default is 'none')
        verbose : boolean
            True/False to determine if output should be printed.
        vertical_symmetry : boolean
            Symmetrize the image in the up/down direction? (The first axis is the vertical axis.)
        horizontal_symmetry: boolean
            Symmetrize the image in the left/right direction?
        integrate : boolean
            Determines if an angular integration should be performed in order to extract the 
            radial intensity distribution. This is useful, for example, for calculating the 
            momentum (speeds) distribution in a photoelectron/photoion angular distribution (PAD)
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
        
        transform_options : tuple
            Additional arguments to be passed to the individual transform functions. 
            See the documentation for the individual transform method for options.
        center_options : tuple
            Additional arguments to be passed to the centering function. 
        
        """
        
        self.IM = IM
                        
        verboseprint = print if verbose else lambda *a, **k: None
        
        if self.IM.ndim == 1 or np.shape(self.IM)[0] <= 2:
                raise ValueError('Data must be 2-dimensional.'
                                 'To transform a single row, use'
                                 'iabel_hansenlaw_transform().')

        if not np.any(use_quadrants): raise ValueError('No image quadrants selected to use')
        
        rows, cols = np.shape(self.IM)
        
        #########################
        # Centering is currently broken pending improvements to the centering functions.
        # see the discussion in Issue 90: https://github.com/PyAbel/PyAbel/issues/90#issuecomment-184301835
        if center == 'none':
            # no centering
            if rows%2 != 1: 
                raise ValueError('Image must have an odd number of columns. Use a centering method.')
            
        elif center == 'com' or center == 'image_center':
            # automatic centering
            raise ValueError('Automatic centering not currently supported.')
            
        else:
            # manual centering
            raise ValueError('Manual centering not currently supported.')
            # set center to y,x value
            
        #########################
        
         
        verboseprint('Calculating {0} Abel transform using {1} method -'.format(direction,method),
                     'image size: {:d}x{:d}'.format(rows, cols))
        
        t0 = time.time()


        # split image into quadrants
        Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(self.IM, reorient=True,
                             vertical_symmetry=vertical_symmetry, horizontal_symmetry=horizontal_symmetry)

        def selected_transform(Z):
            if method == 'hansenlaw':
                if direction == 'forward':
                    return abel.hansenlaw.fabel_hansenlaw(Z, *transform_options)
                elif direction == 'inverse':
                    return abel.hansenlaw.iabel_hansenlaw(Z, *transform_options)
            
            elif method == 'three_point':
                if direction == 'forward':
                    raise ValueError('Forward three-point not implemented')
                elif direction == 'inverse':
                    return abel.three_point.iabel_three_point_transform(Z, *transform_options)
            
            elif method == 'basex':
                if direction == 'forward':
                    raise ValueError('Forward basex not implemented')
                elif direction == 'inverse':
                    warnings.warn('BASEX implementation not working correctly within abel.iabel!')
                    return abel.basex.iabel_basex(Z, *transform_options)
            
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
        self.transform = abel.tools.symmetry.put_image_quadrants((AQ0, AQ1, AQ2, AQ3), odd_size=cols % 2,
                                    vertical_symmetry=vertical_symmetry,
                                    horizontal_symmetry=horizontal_symmetry)

        verboseprint("{:.2f} seconds".format(time.time()-t0))
        
    
    def angular_integration(self,*args,**kwargs):
        self.radial_intensity, self.radial_coordinate = abel.tools.vmi.calculate_speeds(self.transform, *args, **kwargs)
        return self.radial_coordinate, self.radial_intensity
        
    def calculate_anisotropy(self):
        print('Anisotropy not yet implemented!')
        return 1
    
    def residual(self):
        print('Not yet implemented')
        # in some cases, like basex, we can recover the projection while we are performing the inverse transform
        # in that case, we can: return self.IM - self.proj 
    
    @classmethod
    def iabel(cls, *args, **kwargs):
        trans=cls(*args, direction='inverse', **kwargs)
        return trans
    
    @classmethod
    def fabel(cls, *args, **kwargs):
        trans=cls(*args, direction='forward', **kwargs)
        return trans

iabel = AbelTransform.iabel
fabel = AbelTransform.fabel


def main():
    import matplotlib.pyplot as plt
    IM0 = abel.tools.analytical.sample_image_dribinski(n=361)
    trans1 = fabel(IM0,method='hansenlaw')
    IM1 = trans1.transform
    trans2 = iabel(IM1,method='three_point')
    IM2 = trans2.transform
    
    fig, axs = plt.subplots(2,3,figsize=(10,6))
    
    axs[0,0].imshow(IM0)
    axs[0,1].imshow(IM1)
    axs[0,2].imshow(IM2)
    
    axs[1,0].plot(*abel.tools.vmi.calculate_speeds(IM0)[::-1])
    axs[1,1].plot(*trans1.angular_integration())
    axs[1,2].plot(*trans2.angular_integration())
    
    plt.show()
    
    
if __name__ == "__main__":
    main()
    
    
