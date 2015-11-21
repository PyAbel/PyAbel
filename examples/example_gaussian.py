import numpy as np
import matplotlib.pyplot as plt
from abel.basex import BASEX

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

center = n//2

# BASEX Transform: 
# Calculate the inverse abel transform for the centered data
recon = BASEX(F_a, center,  n=501, basis_dir='./', dr=dr,
        verbose=True, calc_speeds=False)

plt.plot(r, recon , '--o',c='red', label='Inverse transform [BASEX]')

ax.legend()

ax.set_xlim(-4,4)
ax.set_ylim(-0.1, 2.7)
ax.set_xlabel('x')
ax.set_ylabel("f(x)")

plt.legend()
plt.show()
