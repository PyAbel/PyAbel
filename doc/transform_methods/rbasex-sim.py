import numpy as np
import matplotlib.pyplot as plt

from pyabel.tools.analytical import SampleImage
from pyabel.tools.vmi import Ibeta
from pyabel.rbasex import rbasex_transform

rmax = 200
scale = 10000

vlim = 0.5
pvlim = 0.5 * np.sqrt(rmax)

def rescaleI(im):
    return np.sqrt(np.abs(im)) * np.sign(im)

# test distribution
source = SampleImage(n=2 * rmax - 1).func / scale
Isrc, _ = Ibeta(source)
Inorm = (Isrc**2).sum()

# simulated projection with Poissonian noise
proj, _ = rbasex_transform(source, direction='forward')
proj[proj < 0] = 0
# (reproducible random noise,
#  see NEP 19: https://numpy.org/neps/nep-0019-rng-policy.html)
proj = np.random.RandomState(0).poisson(proj)

# plot...
fig = plt.figure(figsize=(7, 3.5), frameon=False)

# test distribution
plt.subplot(121)

fig = plt.imshow(rescaleI(source), vmin=-vlim, vmax=vlim, cmap='bwr')

plt.axis('off')
plt.text(0, 2 * rmax, 'test source', va='top')

# simulated projection
plt.subplot(122)

fig = plt.imshow(rescaleI(proj), vmin=0, vmax=pvlim, cmap='gray_r')

plt.axis('off')
plt.text(0, 2 * rmax, 'simulated projection', va='top')

# finish
plt.subplots_adjust(left=0, right=0.97, wspace=0.1,
                    bottom=0.08, top=0.98, hspace=0.5)

#plt.savefig('rbasex-sim.svg')
#plt.show()
