import numpy as np
from scipy.linalg import solve_banded

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pyabel.tools.polynomial import PiecewisePolynomial as PP

n = 10
r0 = 2.3

# radial array with 20 point per pixel and half-integer point split into
# ±ε pairs for sharp transitions for degree = 0
r = np.linspace(0, n - 1, (n - 1) * 20 + 1) + 1e-5
r = np.sort(np.concatenate((r, 0.5 + np.arange(0, n - 1) - 1e-5)))


def f(x):
    return np.exp(-(x - r0)**2 / 2) + np.exp(-(-x - r0)**2 / 2) + \
           np.exp(-(x - 7)**2 * 3)


def p(degree, j):
    if degree == 0:
        return PP(r, [(j - 1/2, j + 1/2, [1], j)])
    elif degree == 1:
        return PP(r, [(j - 1, j, [1,  1], j),
                      (j, j + 1, [1, -1], j)])
    elif degree == 2:
        return PP(r, [(j - 1,   j - 1/2, [0, 0,  2], j - 1),
                      (j - 1/2, j + 1/2, [1, 0, -2], j),
                      (j + 1/2, j + 1,   [0, 0,  2], j + 1)])
    else:  # degree == 3
        return PP(r, [(j - 1, j, [1, 0, -3, -2], j),
                      (j, j + 1, [1, 0, -3,  2], j)])


def q(j):
    return PP(r, [(j - 1, j, [0, 1,  2, 1], j),
                  (j, j + 1, [0, 1, -2, 1], j)])


plt.figure(figsize=(5, 7))

colors = cm.rainbow(np.linspace(1, 0, n - 1))

for degree in range(4):
    plt.subplot(4, 1, 1 + degree)
    plt.title(f'{degree = }',
              fontdict={'fontsize': plt.rcParams['axes.labelsize'],
                        'fontweight': 'bold'},
              loc='left')

    F = f(np.arange(n))
    if degree == 3:
        D = np.concatenate(([0], 3 * (F[2:] - F[:-2]), [0]))
        G = solve_banded((1, 1), ([0, 0] + [1] * (n - 2),
                                  [4] * n,
                                  [1] * (n - 2) + [0, 0]), D)

    total = np.zeros_like(r)
    for j, c in enumerate(colors):
        p_j = F[j] * p(degree, j).func
        plt.fill_between(r, 0, p_j, color=c, alpha=0.5)
        total += p_j
        if degree == 3:
            q_j = G[j] * q(j).func
            plt.plot(r, q_j, c=c)
            total += q_j

    plt.plot(r, f(r), c='k', label='test')
    plt.plot(r, total, '--', c='k', lw=2, label='approx.')

    plt.xlim((0, n - 1))
    if degree == 3:
        plt.xlabel('radius')

    plt.ylim((-0.15, 1.05))
    plt.ylabel('value')

    if degree == 0:
        plt.legend(loc='upper center', bbox_to_anchor=(0.55, 1))

plt.tight_layout()

#plt.savefig('daun-basis.svg')
#plt.show()
