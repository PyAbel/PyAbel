import numpy as np
import matplotlib.pyplot as plt
import abel

# This example performs a BASEX transform of a simple 1D Gaussian function and compares
# this to the analytical inverse Abel transform 

fig, ax= plt.subplots(1,1)
plt.title('Abel tranforms of a gaussian function')

# Analytical inverse Abel: 
n = 101
r_max = 20
sigma = 10

ref = abel.tools.analytical.GaussianAnalytical(n, r_max, sigma,symmetric=False)

ax.plot(ref.r, ref.func, 'b', label='Original signal')
ax.plot(ref.r, ref.abel, 'r', label='Direct Abel transform x0.05 [analytical]')

center = n//2

# BASEX Transform: 
# Calculate the inverse abel transform for the centered data
recon = abel.basex.basex_transform(ref.abel, verbose=True, basis_dir=None, 
        dr=ref.dr, direction='inverse')

ax.plot(ref.r, recon , 'o',color='red', label='Inverse transform [BASEX]', ms=5, mec='none',alpha=0.5)

ax.legend()

ax.set_xlim(0,20)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')

plt.legend()
plt.show()
