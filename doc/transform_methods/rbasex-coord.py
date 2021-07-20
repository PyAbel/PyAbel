from __future__ import print_function, division

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams['lines.solid_capstyle'] = 'round'

xmax, ymax, zmax = 6, 7, 6  # axes
x, y, z = 3, 6, 5  # vector
lo = 0.5  # label offset
an = 11  # angle arc segments
anc = an // 2  # index of central point

fig = plt.figure(figsize=(3, 3), frameon=False)
ax = fig.gca(projection='3d')
ax.set_proj_type('ortho')
ax.view_init(elev=30, azim=45)
ax.set_axis_off()
ax.set_xlim(0, zmax)
ax.set_ylim(0, xmax)
ax.set_zlim(0, ymax)

sax = {'color': 'k', 'lw': 1, 'arrow_length_ratio': 0.1}  # axes style
sb = {'c': 'k', 'lw': 0.5}  # "box" style
sv = {'c': 'k', 'lw': 2}  # vectors style
sa = {'c': 'k', 'lw': 1}  # angle arcs style
sl = {'ha': 'center', 'va': 'center'}  # label style


# axes
ax.quiver(0, 0, 0, 0, xmax, 0, **sax); ax.text(lo, xmax, 0, '$x$', **sl)
ax.quiver(0, 0, 0, 0, 0, ymax, **sax); ax.text(lo, 0, ymax, '$y$', **sl)
ax.quiver(0, 0, 0, zmax, 0, 0, **sax); ax.text(zmax, lo, 0, '$z$', **sl)


# "box"...
ax.plot([0, 0], [0, x], [y, y], **sb)
ax.plot([z, z], [0, x], [y, y], **sb)
ax.plot([z, z], [0, x], [0, 0], **sb)

ax.plot([0, 0], [x, x], [0, y], **sb)
ax.plot([z, z], [x, x], [0, y], **sb)
ax.plot([z, z], [0, 0], [0, y], **sb)

ax.plot([0, z], [0, 0], [y, y], **sb)
ax.plot([0, z], [x, x], [y, y], 'k--')  # "projection"
ax.plot([0, z], [x, x], [0, 0], **sb)

ax.plot([0, z], [0, x], [0, 0], **sb)
ax.plot([0, z], [0, x], [y, y], **sb)


# 3D vector...
ax.plot([0, z], [0, x], [0, y], **sv); ax.scatter(z, x, y, c='k')
# rho
ax.text(z / 2 + lo / 1.5, x / 2, y / 2 - lo / 1.5, '$\\rho$', **sl)

# theta'
xz = np.sqrt(x**2 + z**2)
t = np.linspace(0, np.arctan2(xz, y), an)
ar = 2  # arc radius
x_, y_, z_ = ar * x / xz * np.sin(t), ar * np.cos(t), ar * z / xz * np.sin(t)
ax.plot(z_, x_, y_, **sa)
ax.text(z_[anc] + lo / 1.5, x_[anc], y_[anc] + lo / 1.5, '$\\theta\'$', **sl)

# phi'
t = np.linspace(0, np.arctan2(z, x), an)
ar = 1  # arc radius
x_, y_, z_ = ar * np.cos(t), 0 * t, ar * np.sin(t)
ax.plot(z_, x_, y_, **sa)
ax.text(z_[anc] + lo, x_[anc] + lo, 0, '$\\varphi\'$', **sl)


# 2D vector...
ax.plot([0, 0], [0, x], [0, y], **sv); ax.scatter(0, x, y, c='k')
# r
ax.text(0, x / 2 + lo, y / 2, '$r$', **sl)

# theta
t = np.linspace(0, np.arctan2(x, y), an)
ar = 1.5  # arc radius
x_, y_, z_ = ar * np.sin(t), ar * np.cos(t), 0 * t
ax.plot(z_, x_, y_, **sa)
ax.text(0, x_[anc] + lo / 3, y_[anc] + lo, '$\\theta$', **sl)


plt.tight_layout(pad=0, rect=(-0.06, -0.1, 1.04, 1))

#plt.savefig('rbasex-coord.svg')
#plt.show()
