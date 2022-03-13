import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig = plt.figure(figsize=(3, 3.8), frameon=False)
ax = plt.gca()
ax.set_aspect('equal')

# horizontal centerline
plt.plot([-1.1, 1.1], [2, 2], c='k', lw=1)

# vertical centerline
plt.axvline(0, c='k', lw=1)

# object
ax.add_patch(Circle((0, 2), 1, ec='k', fill=False, lw=2))
ax.text(-0.5, 2, 'Object, $f(r)$\n', ha='center', va='bottom')

# path of integration
plt.axvline(0.5, c='k', lw=1)
ax.text(0.5, 3, ' Path of\n integration', va='center')

# r arrow
ax.annotate('', xy=(0, 2), xytext=(0.5, 2.6),
            arrowprops=dict(arrowstyle='<-', shrinkA=0, shrinkB=0))
ax.text(0.25, 2.3, '$r$', ha='right', va='bottom')

# horizontal axis
ax.annotate('$R$', va='center', xy=(-1.1, 0), xytext=(1.2, 0),
            arrowprops=dict(arrowstyle='<-'))

# projection
R = np.linspace(0, 1, 100)
Rr = 1 - R
g = 1.5 * (Rr - 3 * Rr**2 + 5 * Rr**3 - 2.5 * Rr**4)
plt.plot(R, g, 'k', lw=2)
ax.text(0, 0.75, 'Projection, $g(R)$  \n(Abel transform)  ',
        ha='right', va='top')

plt.axis('off')
plt.xlim((-1.1, 1.3))
plt.ylim((0, 3.2))
plt.subplots_adjust(0.02, 0.05, 0.98, 0.98)

#plt.savefig('hansenlaw-proj.svg')
#plt.show()
