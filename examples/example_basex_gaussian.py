import numpy as np
import matplotlib.pyplot as plt
from abel.basex import BASEX
from abel.analytical import GaussianAnalytical

# This example performs a BASEX transform of a simple 1D Gaussian function and compares
# this to the analytical inverse Abel transform 

fig, ax= plt.subplots(1,1)
plt.title('Abel tranforms of a gaussian function')

# Analytical inverse Abel: 
n = 101
r_max = 20
sigma = 10

ref = GaussianAnalytical(n, r_max, sigma)

ax.plot(ref.r, ref.func, 'b', label='Original signal')

ax.plot(ref.r, ref.abel*0.05, 'r', label='Direct Abel transform x0.05 [analytical]')

center = n//2

# BASEX Transform: 
# Calculate the inverse abel transform for the centered data
recon = BASEX(ref.abel, center,  n=n, basis_dir=None, dr=ref.dr,
        verbose=True, calc_speeds=False)

plt.plot(ref.r, recon , '--o',c='red', label='Inverse transform [BASEX]')

ax.legend()

ax.set_xlim(-20,20)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

plt.legend()
plt.show()
