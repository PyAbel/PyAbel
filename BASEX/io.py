#!/usr/bin/python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os.path

import numpy as np


# The following functions are just conveinent functions for loading and saving images.
# Often you can just use plt.imread('my_file.png') to load a file.
# plt.imread also works for 16-bit tiff files.
def load_raw(filename,start=2,end=1440746,height=1038,width=1388):
    """
     This loads one of the raw VMI images from Vrakking's "VMI_Acquire" software
     It ignores the first two values (which are just the dimensions of the image,
     and not actual data) and cuts off about 10 values at the end.
     I don't know why the files are not quite the right size, but this seems to work.
    """
    
    # Load raw data
    A = np.fromfile(filename, dtype='int32', sep="")
    # Reshape into a numpy array
    return A[start:end].reshape([height, width])
    

def save16bitPNG(filename,data):
    """ It's not easy to save 16-bit images in Python. Here is a way to save a 16-bit PNG
     Again, this is thanks to stackoverflow: #http://stackoverflow.com/questions/25696615/can-i-save-a-numpy-array-as-a-16-bit-image-using-normal-enthought-python
     This requires pyPNG
    """
    
    import png
    with open(filename, 'wb') as f:
        writer = png.Writer(width=data.shape[1], height=data.shape[0], bitdepth=16, greyscale=True)
        data_list = data.tolist()
        writer.write(f, data_list)


def parse_matlab(basename='basis1000', base_dir='./', gzip=False):
    """ Parse matlab basis files, in the format,
            basis1000_1.bst.gz
            basis1000pr_1.bst.gz
    """

    if gzip:
        ext = ".bst.gz"
    else:
        ext = ".bst"
    M = np.loadtxt(os.path.join(base_dir, basename+'pr_1'+ext))
    Mc = np.loadtxt(os.path.join(base_dir, basename+'_1'+ext))
    return M.view(np.matrix), Mc.view(np.matrix)
