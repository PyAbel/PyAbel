# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from .analytical import SymStep

class SymStepBenchmark(object):
    def __init__(self, n, r_max, r1, r2, A0=1.0):
        """
        A benchmark for testing inverse Abel implementations using a
        a symmetric step function that has a known Abel transform

        see https://github.com/PyAbel/PyAbel/pull/16

         Parameters:
             same as for abel.analytical.SymStep
        """

        self.step = SymStep(n, r_max, r1, r2, A0)

    def run(self, recon, ratio=1.0):
        """
        Parameters:
          - recon: 1D ndarray: a reconstruction (i.e. inverse abel) of the original signal
          - ratio: float: in the benchmark take only the central ratio*100% of the step
                                         (exclude possible artefacts on the edges)
        """
        st = self.step
        mask2 = np.abs(np.abs(st.r)- 0.5*(st.r1 + st.r2)) < ratio*0.5*(st.r2 - st.r1)
        err = st.func[mask2]/recon[mask2]
        return np.mean(err), np.std(err), np.sum(mask2)

