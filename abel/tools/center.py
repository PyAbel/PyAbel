import numpy as np

from scipy.ndimage import center_of_mass


def find_center_by_center_of_mass(data, verbose=True, **kwargs):
    com = center_of_mass(data)
    com_round = np.around(com)
    if verbose:
        print("Center of mass at ({0}, {1}) ... round to ({2}, {3})".format(com[1], com[0], com_round[1], com_round[0]))
    return (int(com_round[1]), int(com_round[0]))


def find_center_by_center_of_image(data, verbose=True, **kwargs):
    return (data.shape[1]//2, data.shape[0]//2)


func_method = {
    "auto": find_center_by_center_of_image,
    "com": find_center_by_center_of_mass,
}


def find_center(data, method="auto", verbose=True, **kwargs):
    return func_method[method](data, verbose=verbose, **kwargs)