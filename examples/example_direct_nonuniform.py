import numpy as np
import matplotlib.pyplot as plt
from abel.tools.analytical import GaussianAnalytical
from abel.direct import direct_transform_new

# a Gaussian sampled on a non-uniform grid (denser in more curved regions)
n = 30
r = np.cumsum(np.sin(np.arange(n) / 5)**2 + 0.5) / 10
r -= r[0]  # (r must start at 0)
f = np.exp(-r**2)

ref = GaussianAnalytical(100, r[-1], symmetric=False)

plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.title('Forward transform')
fabel = direct_transform_new(f, r=r, direction='forward')
plt.xlabel('Radius')
plt.ylabel('Intensity')
plt.vlines(r, 0, fabel, lw=0.5, colors='0.8')
plt.plot(r, f, 'C0.-', label='Sampled function')
plt.plot(r, fabel, 'C1.-', label='Numerical transform')
plt.plot(ref.r, ref.abel, 'k:', label='Analytical transform')
plt.legend()

plt.subplot(122)
plt.title('Inverse transform')
f *= ref.abel[0]
iabel = direct_transform_new(f, r=r, direction='inverse')
plt.xlabel('Radius')
plt.vlines(r, 0, f, lw=0.5, colors='0.8')
plt.plot(r, f, 'C0.-', label='Sampled function')
plt.plot(r, iabel, 'C1.-', label='Numerical transform')
plt.plot(ref.r, ref.func, 'k:', label='Analytical transform')
plt.legend()

plt.tight_layout()
plt.show()
