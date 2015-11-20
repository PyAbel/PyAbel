# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


def absolute_ratio_benchmark(analytical, recon):
    """
    Check the absolute ratio between an analytical function and the result
     of a inv. Abel reconstruction.

    Parameters
    ----------
      - analytical: one of the classes from abel.analytical, initialized
      - recon: 1D ndarray: a reconstruction (i.e. inverse abel) given by some PyAbel implementation
    """
    mask = analytical.mask_valid
    err = analytical.func[mask]/recon[mask]
    return np.mean(err), np.std(err), np.sum(mask)

