# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import time

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
    
    def __init__(self,IM,direction=None,method='three_point',center=(None,None),verbose=True,
                 vertical_symmetry=True,horizontal_symmetry=True,use_quadrants=(True,True,True,True),
                 integrate=True,transform_options=()):
        """__init__

        This performs the forward or reverse Abel transform using a user-selected method.
        
        Transform Methods
        --------
        # here we should provide a description of the various methods, complete with references.
       

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
            This specifies the image center in (x,y) format (if a tuple is provided)
            or, if a string is provided, an automatic centering algorithm is used

        """
    
                        
        verboseprint = print if verbose else lambda *a, **k: None
        
        self.IM = IM
    
        if IM.ndim == 1 or np.shape(IM)[0] <= 2:
                raise ValueError('Data must be 2-dimensional.'
                                 'To transform a single row, use'
                                 'iabel_hansenlaw_transform().')

        if not np.any(use_quadrants): raise ValueError('No image quadrants selected to use')
        
        rows, cols = np.shape(IM)
        
        verboseprint('Calculating {0} Abel transform using {1} method -'.format(direction,method),
                     'image size: {:d}x{:d}'.format(rows, cols))
        
        t0 = time.time()
        
        # add code to center the image here!!

        # split image into quadrants
        Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(IM, reorient=True,
                             vertical_symmetry=vertical_symmetry, horizontal_symmetry=horizontal_symmetry)

        def selected_transform(IM):
            
            if method == 'hansenlaw':
                if direction == 'forward':
                    return abel.hansenlaw.fabel_hansenlaw_transform(IM, *transform_options)
                elif direction == 'inverse':
                    return abel.hansenlaw.iabel_hansenlaw_transform(IM, *transform_options)
            
            elif method == 'three_point':
                if direction == 'forward':
                    raise ValueError('Forward three-point not implemented')
                elif direction == 'inverse':
                    return abel.three_point.iabel_three_point_transform(IM, *transform_options)
            
            elif method == 'basex':
                if direction == 'forward':
                    raise ValueError('Forward basex not implemented')
                elif direction == 'inverse':
                    raise ValueError('Coming soon...')
            
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
        return cls(*args,direction='inverse', **kwargs)
    
    @classmethod
    def fabel(cls, *args, **kwargs):
        return cls(*args,direction='forward', **kwargs)


iabel = AbelTransform.iabel
fabel = AbelTransform.fabel
        
def main():
    import matplotlib.pyplot as plt
    IM0 = abel.tools.analytical.sample_image_dribinski(n=361)
    trans1 = fabel(IM0,method='hansenlaw')
    IM1 = trans1.transform
    trans2 = iabel(IM1)
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
    
    
