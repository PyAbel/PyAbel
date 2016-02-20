# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import numpy as np


# The following are just convenient functions for loading and saving images.
# Often you can just use plt.imread('my_file.png') to load a file.
# plt.imread also works for 16-bit tiff files.

def load_raw(filename, start=2, end=1440746, height=1038, width=1388):
    """
    This loads raw VMI images from Vrakking's "VMI_Acquire" software
    It ignores the first two values
    (which are just the dimensions of the image, and not actual data)
    and cuts off about 10 values at the end.
    It's unclear why the files are not quite the right size,
    but this seems to work.
    """

    # Load raw data
    A = np.fromfile(filename, dtype='int32', sep="")
    # Reshape into a numpy array
    return A[start:end].reshape([height, width])


def save16bitPNG(filename, data):
    """
    It's not easy to save 16-bit images in Python.
    Here is a way to save a 16-bit PNG
    Again, this is thanks to stackoverflow:
    http://stackoverflow.com/questions/25696615/\
    can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python
    This requires pyPNG
    """

    import png
    with open(filename, 'wb') as f:
        writer = png.Writer(width=data.shape[1], height=data.shape[0],
                            bitdepth=16, greyscale=True)
        data_list = data.tolist()
        writer.write(f, data_list)


def parse_matlab_basis_sets(path):
    """
    Load matlab generated basis sets files,
    The expected format for the `path` argument is a string of the form
    "some_basis_set_{}_1.bsc" where "{}" will be replaced by "" for
    the first file and "pr" for the second. Gzip compressed text files
    are accepted. For instance:

        basis1000_1.bst.gz

        basis1000pr_1.bst.gz
    """
    M = np.loadtxt(path.format('pr'))
    Mc = np.loadtxt(path.format(''))
    return M, Mc
