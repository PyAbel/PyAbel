# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import matplotlib.pyplot as plt
from time import time
import sys

from abel.direct import fabel_direct, iabel_direct
from abel.tools.analytical import GaussianAnalytical


n = 101
r_max = 30
sigma = 10

ref = GaussianAnalytical(n, r_max, sigma, symmetric=False)

fig, ax = plt.subplots(1,2)

ax[0].set_title('Forward transform of a Gaussian')
ax[1].set_title('Inverse transform of a Gaussian')

ax[0].plot(ref.r, ref.abel, 'b', label='Analytical transform')

recon = fabel_direct(ref.func, dr=ref.dr, correction=True, backend='C')
ax[0].plot(ref.r, recon , '--o',c='red', label='direct')
recon = fabel_direct(ref.func, dr=ref.dr, correction=True, backend='Python')
ax[0].plot(ref.r, recon , ':d', c='k', label='direct naive')


ax[1].plot(ref.r, ref.func, 'b', label='Original function')

recon = iabel_direct(ref.abel, dr=ref.dr, correction=True)
ax[1].plot(ref.r, recon , '--o', c='red', label='direct')
recon = iabel_direct(ref.abel, dr=ref.dr, correction=False)
ax[1].plot(ref.r, recon , ':d', c='k', label='direct - naive')

for axi in ax:
    axi.set_xlim(0, 20)
    axi.legend()

plt.show()

