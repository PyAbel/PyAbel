from __future__ import print_function, division

import matplotlib.pyplot as plt

R = 4

fig = plt.figure(figsize=(4, 2), frameon=False)

ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim((0, 2 * R))
plt.xticks([0, R - 1, R, R + 1], ['$0$', '$R - 1$', '$R$', '$R + 1$'])

plt.ylim((0, 1.1))
plt.yticks([0, 1], ['0', '1'])

plt.vlines([R], 0, 1, color='lightgray')
plt.hlines([1], 0, R, color='lightgray')

plt.plot([0, R - 1, R, R + 1, 2 * R],
         [0,     0, 1,     0,     0],
         'k')

plt.tight_layout()

#plt.savefig('rbasex-bR.svg')
#plt.show()
