# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import time

class abel_transform(object):
    def __init__(self,IM,direction=None,method='three_point',integrate=True,verbose=True,
                 vertical_symmetry=True,horizontal_symmetry=True,use_quadrants=(True,True,True,True),
                 transform_options=()):
                        
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

        # split image into quadrants
        Q0, Q1, Q2, Q3 = abel.tools.symmetry.get_image_quadrants(IM, reorient=True,
                             vertical_symmetry=vertical_symmetry, horizontal_symmetry=horizontal_symmetry)

        def selected_transform(IM):
            
            if method == 'hansenlaw':
                if direction == 'forward':
                    return abel.hansenlaw.fabel_hansenlaw_transform(IM, *transform_options)
                if direction == 'inverse':
                    return abel.hansenlaw.iabel_hansenlaw_transform(IM, *transform_options)
            
            if method == 'three_point':
                if direction == 'forward':
                    raise ValueError('Forward three-point not implemented')
                if direction == 'inverse':
                    return abel.three_point.iabel_three_point_transform(IM, *transform_options)
            
            if method == 'basex':
                if direction == 'forward':
                    raise ValueError('Forward basex not implemented')
                if direction == 'inverse':
                    raise ValueError('Coming soon...')
            
            if method == 'direct':
                if direction == 'forward':
                    raise ValueError('Coming soon...')
                if direction == 'inverse':
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
        
        
        
class fabel(abel_transform):
    def __init__(self, *args, **kwargs):
        super(fabel, self).__init__(*args,direction='forward', **kwargs)

class iabel(abel_transform):
    def __init__(self, *args, **kwargs):
        super(iabel, self).__init__(*args,direction='inverse', **kwargs)
        
        
        
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
    
    
