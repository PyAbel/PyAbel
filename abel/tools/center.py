import numpy as np
from scipy.special import round
from scipy.ndimage import center_of_mass
from .math import fit_gaussian


def find_center_by_center_of_mass(data, verbose=True, **kwargs):
    com = center_of_mass(data)
    com_round = (int(round(com[1])), int(round(com[0])))
    if verbose:
        print("Center of mass at ({0}, {1}) ... round to ({2}, {3})".format(com[1], com[0], com_round[1], com_round[0]))
    return com_round


def find_center_by_center_of_image(data, verbose=True, **kwargs):
    return (data.shape[1] // 2, data.shape[0] // 2)


def find_center_by_fit_gaussian(data, verbose=True, **kwargs):
    x = np.sum(data, axis=0)
    y = np.sum(data, axis=1)
    xc = fit_gaussian(x)[1]
    yc = fit_gaussian(y)[1]
    center = (int(round(xc)), int(round(yc)))
    if verbose:
        print("Gaussian center at ({0}, {1}) ... round to ({2}, {3})".format(xc, yc, center[1], center[0]))
    return center


func_method = {
    "auto": find_center_by_center_of_image,
    "com": find_center_by_center_of_mass,
    "gaussian": find_center_by_fit_gaussian,
}


def find_center(data, method="auto", verbose=True, **kwargs):
    return func_method[method](data, verbose=verbose, **kwargs)
