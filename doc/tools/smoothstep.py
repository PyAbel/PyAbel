from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

n = 12
rmin = 2.5
rmax = 9.5
w = 1

r = np.linspace(0, n, n * 20)


def smoothstep(r):
    if r < rmin - w or r > rmax + w:
        return 0
    elif r < rmin + w:
        t = (r - (rmin - w)) / (2 * w)
        return t**2 * (3 - 2 * t)
    elif r > rmax - w:
        t = ((rmax + w) - r) / (2 * w)
        return t**2 * (3 - 2 * t)
    else:  # rmin + w < r < rmax + w
        return 1


fig = plt.figure(figsize=(6, 2), frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim((0, n))
plt.xticks([0, rmin, rmax], ['$0$', r'$r_{\rm min}$', r'$r_{\rm max}$'])

plt.ylim((0, 1.1))
plt.ylim(bottom=0)
plt.yticks([0, 0.5, 1], ['$0$', '$A/2$', '$A$'])

plt.vlines([rmin - w, rmin, rmin + w], 0, 1, color='lightgray')
plt.vlines([rmax - w, rmax, rmax + w], 0, 1, color='lightgray')
plt.hlines([0.5, 1], 0, rmax + w, color='lightgray')

textprm = {'horizontalalignment': 'center',
           'verticalalignment': 'bottom'}
plt.text(rmin - w/2, 1, '$w$', textprm)
plt.text(rmin + w/2, 1, '$w$', textprm)
plt.text(rmax - w/2, 1, '$w$', textprm)
plt.text(rmax + w/2, 1, '$w$', textprm)

plt.plot(r, map(smoothstep, r), color='red')

plt.tight_layout(pad=1)

#plt.show()
#plt.savefig('smoothstep.svg')
