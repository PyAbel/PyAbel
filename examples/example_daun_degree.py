# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, splev
import abel

# one quadrant of Dribinski sample image, its size and intensity distribution
im = abel.tools.analytical.SampleImage().image
q0 = abel.tools.symmetry.get_image_quadrants(im)[0]
n = q0.shape[0]
I0, _ = abel.tools.vmi.Ibeta(q0, origin='ll')

# forward-transformed quadrant
Q = abel.rbasex.rbasex_transform(q0, origin='ll', direction='forward')[0]
# rescale intensities to 1000 max
scale = 1000 / Q.max()
q0 *= scale
I0 *= scale
Q *= scale
# add Poissonian noise (comment out to compare the clean transform results)
Q = np.random.RandomState(0).poisson(Q)

# array for corresponding intensity distributions
Is = []

plt.figure(figsize=(7, 5))

# transformed images
for degree in range(4):
    plt.subplot(3, 4, 1 + degree)
    plt.axis('off')

    q = abel.daun.daun_transform(Q, degree=degree)
    plt.imshow(q, clim=(-3, 3), cmap='seismic')
    plt.text(n / 2, 0, 'degree=' + str(degree), ha='center', va='top')

    I, _ = abel.tools.vmi.Ibeta(q, origin='ll')
    Is.append(I)

# pixel subdivisions for smooth plots
sub = 10
rsub = np.linspace(0, n, sub * n + 1)

# plots of intensity distributions
for plot in [2, 3]:
    plt.subplot(3, 1, plot)
    plt.axhline(0, c='k', ls=':', lw=1)
    plt.plot(I0, '--k', lw=1)

    # degree = 0: plot with steps
    plt.step(np.arange(n), Is[0], lw=1, label='0', where='mid')
    # degree = 1: plot with lines
    plt.plot(Is[1], lw=1, label='1')
    # degree = 2: plot using parabolic segments
    r1, r2 = np.arange(sub // 2) / sub, np.arange(sub // 2, 0, -1) / sub
    b = np.concatenate((2 * r1**2, 1 - 2 * r2**2, 1 - 2 * r1**2, 2 * r2**2))
    I = np.zeros_like(rsub)
    for m in range(1, n):
        i0 = sub * m
        I[i0 - sub: i0 + sub] += Is[2][m] * b
    plt.plot(rsub, I, lw=1, label='2')
    # degree = 3: plot with cubic splines
    spl = make_interp_spline(np.arange(n), Is[3], bc_type='clamped')
    plt.plot(rsub, splev(rsub, spl), lw=1, label='3')

    plt.xlim((0, n - 1))
    plt.yticks([])
    if plot == 2:  # full y range
        plt.xticks([])
        plt.legend(loc='upper center', bbox_to_anchor=(0.25, 1))
    else:  # magnified
        plt.ylim((-2000, 8000))

plt.subplots_adjust(left=0.01, right=0.98,
                    bottom=0.05, top=1,
                    wspace=0, hspace=0.03)
plt.show()
