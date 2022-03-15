import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, Rectangle

fig = plt.figure(figsize=(4, 3.2), frameon=False)
ax = plt.gca()
ax.set_aspect('equal')

# sphere tilt angle
theta = np.pi / 6
theta_deg = theta / np.pi * 180
st = np.sin(theta)
ct = np.cos(theta)

# properties for arrows
arr_prop = dict(arrowstyle='<-', shrinkA=0, shrinkB=0, lw=1)
arr_r = 1.4
arr_x = arr_r * st
arr_y = arr_r * ct

# 3D Newton sphere
ax.add_patch(Circle((-2, -2), 1, ec='k', fill=False, lw=2, clip_on=False))
ax.annotate('', xy=(arr_x - 2, -arr_y - 2), xytext=(-arr_x - 2, arr_y - 2),
            arrowprops=arr_prop)
ax.text(-arr_x - 2, arr_y - 2, '$z\'\\,$ ', ha='right', va='top')
ax.add_patch(Arc((-2, -2), 2, 0.5, angle=theta_deg, theta1=180, theta2=0,
             ec='k', fill=False, lw=1))
ax.add_patch(Arc((-2, -2), 2, 0.5, angle=theta_deg, theta1=0, theta2=180,
             ec='k', ls='--', fill=False, lw=1))
ax.annotate('', xy=(-2, -2), xytext=(st - 2, ct - 2), arrowprops=arr_prop)
ax.text(st - 2, ct - 2, '$r_k$   ', ha='right', va='top')

# 3D -> 2D
sc = np.sqrt(0.5)
ax.plot([-sc - 2, -sc], [sc - 2, sc], ':k', lw=1)
ax.plot([sc - 2, sc], [-sc - 2, -sc], ':k', lw=1)

# detector plane
ax.add_patch(Rectangle((-1.3, -1.3), 2.8, 2.8, ec='k', fill=False, lw=1))
ax.annotate('', xy=(-arr_r + 0.2, 0), xytext=(arr_r, 0), arrowprops=arr_prop)
ax.text(arr_r, 0, '\n$x$', ha='right', va='center', linespacing=2)
ax.annotate('', xy=(0, -arr_r + 0.2), xytext=(0, arr_r), arrowprops=arr_prop)
ax.text(0, arr_r, ' $\\,z$', ha='left', va='top')
ax.annotate('', xy=(arr_x, -arr_y), xytext=(-arr_x, arr_y), arrowprops=arr_prop)
ax.text(-arr_x, arr_y, '$z\'\'\\,$ ', ha='right', va='top')
ax.add_patch(Arc((0, 0), 1, 1, angle=90, theta2=theta_deg,
             ec='k', fill=False, lw=1))
ax.text(-st / 4, 0.5, '$\\theta$ ', ha='center', va='bottom')

# 2D projection
ax.add_patch(Circle((0, 0), 1, ec='k', fill=False, lw=2))
ax.plot([-ct, ct], [-st, st], 'k', lw=1)

# 2D -> 1D
ax.plot([0, 2], [1, 1], ':k', lw=1)
ax.plot([0, 2], [-1, -1], ':k', lw=1)

# 1D projection
ax.annotate('', xy=(2, 0), xytext=(3, 0), arrowprops=arr_prop)
ax.text(3, 0, '$L(z)$\n', ha='right', va='center', linespacing=2)
ax.annotate('', xy=(2, -arr_r + 0.2), xytext=(2, arr_r), arrowprops=arr_prop)
ax.text(2, arr_r, '$z\\,$ ', ha='right', va='top')
z = np.linspace(-1, 1, 50)
ax.plot([2] + list(2.25 + z**2 / 4) + [2],
        [-1] + list(z) + [1], 'k', lw=2)

plt.axis('off')
plt.xlim((-3, 3))       # 6
plt.ylim((-3.25, 1.5))  # 4.75
plt.subplots_adjust(0.02, 0.02, 0.98, 0.98)

#plt.savefig('linbasex-proj.svg')
#plt.show()
