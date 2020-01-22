from __future__ import division, print_function

import numpy as np
from scipy.linalg import inv, svd
import matplotlib.pyplot as plt

from abel.rbasex import _bs_rbasex

Rmax = 40

# SVD for 0th-order inverse Abel transform
P, = _bs_rbasex(Rmax, 0, False)
A = inv(P.T)
V, s, UT = svd(A)

# setup x axis
def setx():
  plt.xlim((0, Rmax))
  plt.xticks([0, 1/4 * Rmax, 1/2 * Rmax, 3/4 * Rmax, Rmax],
             ['$0$', '', '', '', '$r_{\\rm max}$'])

# plot i-th +- 0, 1 singular vectors
def plotu(i, title):
  plt.title('$\\mathbf{v}_i,\\quad i = ' + title + '$')
  i = int(i)
  plt.plot(V[:, i - 1], '#DD0000')
  plt.plot(V[:, i], '#00AA00')
  plt.plot(V[:, i + 1], '#0000FF')
  setx()

fig = plt.figure(figsize=(6, 6), frameon=False)

# singular values
plt.subplot(321)
plt.title('$\\sigma_i$')
plt.plot(s, 'k')
setx()
plt.ylim(bottom=0)

# vectors near 0
plt.subplot(322)
plotu(1, '0, 1, 2')

# vectors near 1/4
plt.subplot(323)
plotu(1/4 * Rmax, '\\frac{1}{4} r_{\\rm max} \\pm 0, 1')

# vectors near middle
plt.subplot(324)
plotu(1/2 * Rmax, '\\frac{1}{2} r_{\\rm max} \\pm 0, 1')

# vectors near 3/4
plt.subplot(325)
plotu(3/4 * Rmax, '\\frac{3}{4} r_{\\rm max} \\pm 0, 1')

# vectors near end
plt.subplot(326)
plotu(Rmax - 1, 'r_{\\rm max} - 2, 1, 0')

plt.tight_layout()

#plt.savefig('rbasex-SVD.svg')
#plt.show()
