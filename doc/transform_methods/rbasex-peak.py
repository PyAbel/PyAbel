from __future__ import print_function, division

import numpy as np
from numpy import log, arctan2
import matplotlib.pyplot as plt

n = 2
R = 6
rmax = 8.5
samp = 20  # samples per 1 px

fig = plt.figure(figsize=(6, 3), frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim((0, rmax))

r = np.linspace(0, rmax, rmax * samp + 1)
r[0] = np.finfo(float).eps  # (to avoid division by 0)

def sqrtp(x):
    return np.sqrt(x, where=x > 0, out=np.zeros_like(x))

def F_1(z, rho):
    return 1/2 * z * rho / r + 1/2 * r * log(z + rho)

def F0(z, rho):
    return z

def F1(z, rho):
    return r * log(z + rho)

def F2(z, rho):
    return r * arctan2(z, r)

F = {-1: F_1, 0: F0, 1: F1, 2: F2}

def p(R, n):
    zRm1 = sqrtp((R - 1)**2 - r**2)
    zR   = sqrtp( R**2      - r**2)
    zRp1 = sqrtp((R + 1)**2 - r**2)

    rhoRm1 = np.maximum(r, R - 1)
    rhoR   = np.maximum(r, R    )
    rhoRp1 = np.maximum(r, R + 1)

    return 4 * (r * F[n - 1](zR, rhoR) - R * F[n](zR, rhoR)) - \
           2 * (r * F[n - 1](zRm1, rhoRm1) - (R - 1) * F[n](zRm1, rhoRm1)) - \
           2 * (r * F[n - 1](zRp1, rhoRp1) - (R + 1) * F[n](zRp1, rhoRp1))

plt.plot([-1, R - 1, R, R + 1, rmax + 1],
         [ 0,     0, 1,     0,        0],
         'ro--', label='1 px')
P = p(R, n)
plt.plot(r, P, 'r')
plt.plot(r[::samp], P[::samp], 'ro:')

plt.plot([-1, R - 2, R - 1,   R, R + 1, R + 2, rmax + 1],
         [ 0,     0,   0.4, 0.6,   0.4,     0,        0],
         'go--', label='3 px')
P = 0.4 * p(R - 1, n) + 0.6 * p(R, n) + 0.4 * p(R + 1, n)
plt.plot(r, P, 'g')
plt.plot(r[::samp], P[::samp], 'go:')

plt.ylim(bottom=0)

plt.legend(markerscale=0)
plt.tight_layout()

#plt.savefig('rbasex-peak.svg')
#plt.show()
