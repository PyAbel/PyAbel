import numpy as np
import scipy.special
from scipy.ndimage import center_of_mass
from .math import fit_gaussian


def find_center_by_center_of_mass(data, verbose=True, **kwargs):
    try:
        round = kwargs["round"]
    except KeyError:
        round = True

    com = center_of_mass(data)
    center = com[1], com[0]
    
    if verbose:
        to_print = "Center of mass at ({0}, {1})".format(center[0], center[1])
    
    if round:
        center = scipy.special.round(center)
        center = (int(center[0]), int(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print) 

    return center


def find_center_by_center_of_image(data, verbose=True, **kwargs):
    return (data.shape[1] // 2, data.shape[0] // 2)


def find_center_by_fit_gaussian(data, verbose=True, **kwargs):
    try:
        round = kwargs["round"]
    except KeyError:
        round = True
    
    x = np.sum(data, axis=0)
    y = np.sum(data, axis=1)
    xc = fit_gaussian(x)[1]
    yc = fit_gaussian(y)[1]
    center = (xc, yc)
    
    if verbose:
        to_print = "Gaussian center at ({0}, {1})".format(center[0], center[1])
    
    if round:
        center = scipy.special.round(center)
        center = (int(center[0]), int(center[1]))
        if verbose:
            to_print += " ... round to ({0}, {1})".format(center[0], center[1])

    if verbose:
        print(to_print) 

    return center

func_method = {
    "auto": find_center_by_center_of_image,
    "com": find_center_by_center_of_mass,
    "gaussian": find_center_by_fit_gaussian,
}


def find_center(data, method="auto", verbose=True, **kwargs):
    return func_method[method](data, verbose=verbose, **kwargs)
