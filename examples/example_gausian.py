import numpy as np
import matplotlib.pyplot as plt
from BASEX import BASEX

fig, ax= plt.subplots(1,1)
plt.title('Abel tranforms of a gaussian function')

# Analytical inverse Abel: 
n = 501
r = np.linspace(-20, 20, n)
dr = np.diff(r)[0]
fr = np.exp(-r**2)
#fr += 1e-1*nprandom.rand(n)
ax.plot(r, fr,'b', label='Original signal')
F_a = (np.pi)**0.5*fr.copy()
ax.plot(r, F_a, 'r', label='Direct Abel transform [analytical]')

# BASEX Transform: 
inv_ab = BASEX(n=501, nbf=250, basis_dir='./', verbose=True, calc_speeds=False)

# Calculate the inverse abel transform for the centered data
center = n//2
recon = inv_ab(F_a, center , median_size=2,
                    gaussian_blur=0, post_median=0)

scale = 1./dr
recon *= scale

print(np.median((fr/recon)[490:-490]))

plt.plot(r, recon , '--o',c='red', label='Inverse transform [BASEX]')

ax.legend()

ax.set_xlim(-4,4)
ax.set_ylim(-0.1, 2.7)
ax.set_xlabel('x')
ax.set_ylabel("f(x)")

plt.legend()
plt.show()
