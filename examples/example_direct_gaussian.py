# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from time import time
import sys

from abel.direct import direct_transform
from abel.tools.analytical import GaussianAnalytical


n = 101
r_max = 30
sigma = 10

ref = GaussianAnalytical(n, r_max, sigma, symmetric=False)

fig, ax = plt.subplots(1,2)

# forward Abel transform
reconC = direct_transform(ref.func, dr=ref.dr, direction="forward",
                          correction=True)
reconP = direct_transform(ref.func, dr=ref.dr, direction="forward",
                          correction=False)

ax[0].set_title('Forward transform of a Gaussian', fontsize='smaller')
ax[0].plot(ref.r, ref.abel, label='Analytical transform')
ax[0].plot(ref.r, reconC , '--', label='correction=True')
ax[0].plot(ref.r, reconP , ':', label='correction=False')
ax[0].set_ylabel('intensity (arb. units)')
ax[0].set_xlabel('radius')


# inverse Abel transform
reconc = direct_transform(ref.abel, dr=ref.dr, direction="inverse",
                          correction=True)
 
reconnoc = direct_transform(ref.abel, dr=ref.dr, direction="inverse",
                         correction=False)

ax[1].set_title('Inverse transform of a Gaussian', fontsize='smaller')
ax[1].plot(ref.r, ref.func, 'C0', label='Original function')
ax[1].plot(ref.r, reconc , 'C1--', label='correction=True')
ax[1].plot(ref.r, reconnoc , 'C2:', label='correction=False')
ax[1].set_xlabel('radius')

for axi in ax:
    axi.set_xlim(0, 20)
    axi.legend(labelspacing=0.1, fontsize='smaller')

plt.savefig("plot_example_direct_gaussian.png", dpi=100)
plt.show()
