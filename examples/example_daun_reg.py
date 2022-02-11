# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import abel

# one quadrant of Dribinski sample image, its size and intensity distribution
im = abel.tools.analytical.SampleImage().func
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
# add Poissonian noise
Q = np.random.RandomState(0).poisson(Q)

# regularization parameters
regs = [None,
        ('diff', 100),  # RMS optimal is ~50
        ('L2c', 50),    # RMS optimal is ~25
        'nonneg']

# array for corresponding intensity distributions
Is = []

plt.figure(figsize=(7, 5))

# transformed images
for i, reg in enumerate(regs):
    plt.subplot(3, 4, 1 + i)
    plt.axis('off')

    q = abel.daun.daun_transform(Q, degree=1, reg=reg)
    plt.imshow(q, clim=(-3, 3), cmap='seismic')
    plt.text(n / 2, 0, 'reg=' + repr(reg), ha='center', va='top')

    I, _ = abel.tools.vmi.Ibeta(q, origin='ll')
    Is.append(I)

# plots of intensity distributions
for plot in [2, 3]:
    plt.subplot(3, 1, plot)
    plt.axhline(0, c='k', ls=':', lw=1)
    plt.plot(I0, '--k', lw=1)

    for i, reg in enumerate(regs):
        plt.plot(Is[i], lw=1, label=repr(reg))

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
