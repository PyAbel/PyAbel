from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

n = 10
s = 1.0

x = np.linspace(0, n - 1, n + (n - 1) * 20)


def rho(r, k):
    if k == 0:
        y = np.exp(-(r/s)**2)
    else:
        e = np.exp(1)
        y = (e/k**2)**(k**2) * (r/s)**(2*k**2) * np.exp(-(r/s)**2)
    return y


plt.figure(figsize=(6, 3))

colors = cm.rainbow(np.linspace(1, 0, 7))

for k, c in enumerate(colors):
    plt.plot(x, rho(x, k), color=c, label=r'$k = {}$'.format(k))

plt.plot(x, sum([rho(x, k) for k in range(n)]),
         color='black', label=r'$\sum\rho_k$')

plt.xlabel('$r$')
plt.xlim((0, n - 1))
plt.grid(axis='x')

plt.ylabel(r'$\rho_k(r)$')
plt.ylim((0, 1.5))
plt.yticks(np.arange(0, 2, 0.5))

plt.legend()
plt.tight_layout(pad=1)

#plt.show()
#plt.savefig('basex-basis.svg')
