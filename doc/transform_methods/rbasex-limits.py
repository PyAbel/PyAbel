import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

R = 5
r = 2.5

zRm1 = np.sqrt((R - 1)**2 - r**2)
zR   = np.sqrt( R**2      - r**2)
zRp1 = np.sqrt((R + 1)**2 - r**2)

fig = plt.figure(figsize=(3.5, 3), frameon=False)

ax = plt.gca()
ax.set_aspect('equal')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlim((0, R + 1.5))
plt.xticks([0, r, R - 1, R, R + 1], ['$0$', '$r$', '$R - 1$', '$R$', '$R + 1$'])

plt.ylim((0, R + 1.5))
plt.yticks([0, zRm1, zR, zRp1], ['$0$', '$z_{R-1}$', '$z_R$', '$z_{R+1}$'])

t = np.linspace(0, np.pi / 2, 30)
xy = np.vstack((np.cos(t), np.sin(t)))
xyRm1 = (R - 1) * xy
xyR   =  R      * xy
xyRp1 = (R + 1) * xy

style = {'closed': False, 'lw': 2, 'fill': False, 'ec': 'lightgray'}
ax.add_patch(Polygon(np.hstack((xyRm1, xyR[::-1])).T, hatch='//', **style))
ax.add_patch(Polygon(np.hstack((xyR, xyRp1[::-1])).T, hatch='\\\\', **style))

plt.plot(xyRm1[0], xyRm1[1], 'k')
plt.plot(xyR[0]  , xyR[1]  , 'k')
plt.plot(xyRp1[0], xyRp1[1], 'k')

plt.plot(r, zRm1, 'ko')
plt.plot(r, zR  , 'ko')
plt.plot(r, zRp1, 'ko')

plt.vlines([r], 0, zRp1, lw=2, linestyles='dashed')
plt.hlines([zRm1, zR, zRp1], 0, r, lw=1)

plt.tight_layout()

#plt.savefig('rbasex-limits.svg')
#plt.show()
