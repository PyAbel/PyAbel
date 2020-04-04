from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

from abel.tools.analytical import SampleImage
from abel.tools.vmi import Ibeta
from abel.rbasex import rbasex_transform

# test distribution
rmax = 200
scale = 10000
source = SampleImage(n=2 * rmax - 1).image / scale
Isrc, _ = Ibeta(source)
Inorm = (Isrc**2).sum()

# simulated projection fith Poissonian noise
proj, _ = rbasex_transform(source, direction='forward')
proj[proj < 0] = 0
proj = np.random.RandomState(0).poisson(proj)  # (reproducible, see NEP 19)

# calculate relative RMS error for given regularization parameters
def rmse(method, strengths=None):
    if strengths is None:
        _, distr = rbasex_transform(proj, reg=method, out=None)
        I, _ = distr.Ibeta()
        return ((I - Isrc)**2).sum() / Inorm

    err = np.empty_like(strengths, dtype=float)
    for n, strength in enumerate(strengths):
        _, distr = rbasex_transform(proj, reg=(method, strength), out=None)
        I, _ = distr.Ibeta()
        err[n] = ((I - Isrc)**2).sum() / Inorm
    return err

# get all data
L2_str = (np.linspace(0, np.sqrt(1000), 51)**2)[:30]
L2_err = rmse('L2', L2_str)

diff_str = np.linspace(0, np.sqrt(1000), 51)**2
diff_err = rmse('diff', diff_str)

pos_err = rmse('pos')

SVD_str = np.arange(0, 0.056, 1.0 / rmax)
SVD_err = rmse('SVD', SVD_str)

# plot...
fig = plt.figure(figsize=(6, 3), frameon=False)

# L2, diff, pos
plt.subplot(121)

plt.plot(np.sqrt(L2_str), L2_err, label='L2')
plt.plot(np.sqrt(diff_str), diff_err, label='diff')
plt.plot([0, np.sqrt(1000)], [pos_err, pos_err], '--', label='pos')

plt.ylim(bottom=0)
plt.ylabel('relative RMS error')

plt.xlim((0, np.sqrt(1000)))
ticks = [0, 30, 100, 300, 1000]
plt.xticks(np.sqrt(ticks), map(str, ticks))
plt.xlabel('strength')

plt.legend()

# SVD
plt.subplot(122)

plt.step(SVD_str, SVD_err, where='mid', c='C3', label='SVD')

plt.yticks([0.079, 0.080])
plt.ylabel('relative RMS error')

plt.xlim((0, 0.055))
plt.xlabel('strength')

plt.legend()

# finish
plt.tight_layout()

#plt.savefig('rbasex-regRMSE.svg')
#plt.show()
