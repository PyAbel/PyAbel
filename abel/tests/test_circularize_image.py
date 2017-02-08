from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import abel
import scipy.interpolate

def scaling(angle, factor=0.1):
    # define a simple scaling that will squish a circle into a "flower"
    return 1 + factor*(np.sin(2*angle)**4)

def sample_image(size=511):

    Z = abel.tools.analytical.sample_image(n=size, name='Ominus', sigma=2)

    # image perturbation ------------
    #define a simple grid
    x = np.linspace(-size*0.5+0.5,size*0.5-0.5,size)
    y = np.linspace(-size*0.5+0.5,size*0.5-0.5,size)
    X,Y = np.meshgrid(x,y)

    # convert to polar coords, R and theta
    R = np.sqrt(X**2+Y**2)
    T = np.arctan2(Y,X)

    Z = Z*R**2 # just making the sample image more beautiful...

    R_rescaled = R*scaling(T) # angle-dependent scaling

    # now, convert our squished angular grid back to a squished X, Y grid
    X_rescaled = R_rescaled*np.cos(T)
    Y_rescaled = R_rescaled*np.sin(T)

    Z_rescaled = scipy.interpolate.griddata( (X_rescaled.ravel(),
                                   Y_rescaled.ravel()), Z.ravel(), (X,Y) )

    return Z, Z_rescaled

def test_circularize_image():

    IM, IMperturb = sample_image(size=511)

    nslices = 32

    IMcirc, angle, scalefactor, spline =\
        abel.tools.circularize.circularize_image(IMperturb,
                   method='lsq', nslices=nslices, zoom=1, smooth=0,
                   return_correction=True)

    diff = (IMcirc - IM).sum(axis=1).sum(axis=0)

    assert int(diff) == 1542426

    assert len(angle) == nslices

    assert int(angle[-1]*100) == 304
    
    assert int(scalefactor[4]*100) == 92


if __name__ == "__main__":
    test_circularize_image()
