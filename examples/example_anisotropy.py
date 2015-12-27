#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from abel.hansenlaw import *
from abel.tools import *
import matplotlib.pylab as plt

# experimental data, O2- photodetachment velocity-map image
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")

# inverse Abel transform
AIM = iabel_hansenlaw(IM)

#
beta, amp, (theta,intensity), (theta_sub,PAD) = anisotropy(AIM, r_range=[(373,385),])

print("anisotropy parameter β = {:.3f}+-{:.3f}".format(beta[0],beta[1]))

plt.plot(theta, intensity,'b',label="O2- r=373:385")
plt.plot(theta_sub, PAD,'r',lw=2,label="fit")
plt.annotate("$\\beta = {:.3f}\pm{:.3f}$".format(beta[0],beta[1]), (-2,-2),
             fontsize='large')
plt.title("O2- electron anisotropy $I {\propto} [1 + \\beta P_{2}(\cos{\\theta})]$")
plt.legend(loc=0)
plt.savefig("example_anisotropy.png")
plt.show()
