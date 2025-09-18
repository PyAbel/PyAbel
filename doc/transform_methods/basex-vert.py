import numpy as np
from scipy.linalg import inv
import matplotlib.pyplot as plt

n = 20
s = 2.0

m = 2 * (n + 1)
M = np.empty((m, n))
for i in range(m):
    for j in range(n):
        M[i, j] = np.exp(-2*(i / s - j)**2)
M = M.dot(inv((M.T).dot(M))).dot(M.T)
xM = np.arange(m)

plt.figure(figsize=(6, 2))

plt.xlim((-15, 15))

plt.axhline(0, color='lightgray')

plt.plot(xM - n,     M[n],   '.-', label='even lines', color='black')
plt.plot(xM - (n+1), M[n+1], '.-', label='odd lines',  color='red')

plt.legend()
plt.tight_layout()

#plt.show()
#plt.savefig('basex-vert.svg')
