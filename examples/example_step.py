
import matplotlib.pyplot as plt
from abel.basex import BASEX
from abel.analytical import SymStep

fig, ax= plt.subplots(1,1)
plt.title('Abel tranforms of a step function')

n = 501
r_max = 50
A0 = 10.0
r1 = 6.0
r2 = 14.0

# define a symmetric step function and calculate its analytical Abel transform
st = SymStep(n, r_max, r1, r2, A0)

ax.plot(st.r, st.func,'b', label='Original signal')

ax.plot(st.r, st.abel*0.05, 'r', label='Direct Abel transform x0.05 [analytical]')

# BASEX Transform: 
inv_ab = BASEX(n=n, basis_dir='./', verbose=True, calc_speeds=False)

# Calculate the inverse abel transform for the centered data
center = n//2
recon = inv_ab(st.abel, center , median_size=2,
                    gaussian_blur=0, post_median=0)

plt.plot(st.r, 10*recon , '--o',c='red', label='Inverse transform x10 [BASEX]')

ax.legend()

ax.set_xlim(-20,20)
#ax.set_ylim(-0.1, 2.7)
ax.set_xlabel('x')
ax.set_ylabel("f(x)")

plt.legend()
plt.show()
