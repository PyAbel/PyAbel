import numpy as np
from scipy.ndimage import center_of_mass
from .math import fit_gaussian
import warnings
import scipy.ndimage

def center_image(data, center = 'com', verbose=False):
    if isinstance(center, str) or isinstance(center, unicode):
        center = find_center(data, center, verbose=verbose)
    
    centered_data = set_center(data, center, verbose=verbose)
    return centered_data

def set_center(data, center, crop='maintain_size', verbose=True):
    c0,c1 = center
    if isinstance(c0,(int,long)) and isinstance(c1,(int,long)):
        warnings.warn('Integer center detected, but not respected.'
                      'treating center as float and interpolating!')
        # need to include code here to treat integer centers
        # probably can use abel.tools.symmetry.center_image_asym(),
        # but this function lacks the ability to set the vertical center
        
    old_shape  = data.shape
    old_center = data.shape[0]/2.0, data.shape[1]/2.0
    delta0 = old_center[0] - center[0] 
    delta1 = old_center[1] - center[1] 
    centered_data = scipy.ndimage.interpolation.shift(data, (delta0,delta1))
    
    if crop == 'maintain_size':
        return centered_data
    elif crop == 'valid_region':
        # crop to region containing data
        raise ValueError('Not implemented')
    elif crop == 'maintain_data':
        # pad the image so that the center can be moved without losing any of the original data
        # we need to pad the image with zeros before using the shift() function
        raise ValueError('Not implemented')

def find_center(data, method='image_center', verbose=True, **kwargs):
    return func_method[method](data, verbose=verbose, **kwargs)

def find_center_by_center_of_mass(data, verbose=True, round_output=False, **kwargs):
    com = center_of_mass(data)
    center = com[1], com[0]
    
    if verbose:
        to_print = "Center of mass at ({0}, {1})".format(center[0], center[1])
    
    if round_output:
        center = (round(center[0]), round(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print) 

    return center


def find_center_by_center_of_image(data, verbose=True, **kwargs):
    return (data.shape[1] // 2 + data.shape[1]%2, data.shape[0] // 2 + data.shape[0]%2)


def find_center_by_gaussian_fit(data, verbose=True, round_output=True, **kwargs):
    x = np.sum(data, axis=0)
    y = np.sum(data, axis=1)
    xc = fit_gaussian(x)[1]
    yc = fit_gaussian(y)[1]
    center = (xc, yc)
    
    if verbose:
        to_print = "Gaussian center at ({0}, {1})".format(center[0], center[1])
    
    if round_output:
        center = (round(center[0]), round(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print) 

    return center

func_method = {
    "image_center": find_center_by_center_of_image,
    "com": find_center_by_center_of_mass,
    "gaussian": find_center_by_gaussian_fit,
}




