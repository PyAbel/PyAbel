import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig = plt.figure(figsize=(3, 3.1), dpi=200, frameon=False)
ax = plt.gca()
ax.set_aspect('equal')

plt.xlim((0, 6))
plt.ylim((0, 6))

# pixel grid
plt.xticks(range(7))
plt.yticks(range(7))
plt.grid(True, c='k', ls=':')

# remove axes ticks and labels
ax.tick_params(axis='both', which='both', length=0)
ax.set_xticklabels([])
ax.set_yticklabels([])

# centerline
plt.axvline(0.5, ls='--', c='b')
plt.annotate('', xy=(0.5, -0.4), xytext=(0.5, -1), annotation_clip=False,
             arrowprops=dict(arrowstyle='->', color='b'))
plt.text(0, -1, 'image center line', color='b', ha='left', va='top')

# pixel indices
for n in range(7):
    plt.text(n + 0.5, 0, '\n' +
             ['0', '1', '...', '...', '$n - 2$', '$n - 1$', 'pixel'][n],
             ha='center', va='center', linespacing=2, color='b')

# row
ax.add_patch(Rectangle((0, 2), 6, 1, ec='k', fill=False, clip_on=False))
for n in range(7):
    plt.text(n + 0.5, 2.5,
             ['$x_{N-1}$', '$x_{N-2}$', '...', '...', '$x_1$', '$x_0$',
              '  image\n  row'][n],
             ha='center', va='center')

plt.subplots_adjust(0.02, 0.19, 0.83, 0.98)

plt.savefig('hansenlaw-recur.svg')
plt.savefig('hansenlaw-recur.pdf')
#plt.show()
