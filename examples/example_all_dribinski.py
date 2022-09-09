# -*- coding: utf-8 -*-

# This example compares some available inverse Abel transform methods
# for the Dribinski sample image

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from time import time
import matplotlib.pylab as plt
import numpy as np

import abel

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# inverse Abel transform methods -----------------------------
transforms = [
    "basex",
    "direct",
    "hansenlaw",
    "linbasex",
    "onion_peeling",
    "rbasex",
    "three_point",
    "two_point",
]
# number of transforms:
ntrans = len(transforms)

# sample image radius in pixels
R = 150

fIM = abel.tools.analytical.SampleImage(n=2 * R + 1, name="Dribinski").abel
# fIM += np.random.normal(0, 5000, fIM.shape)  # try adding some noise

print("image shape {}".format(fIM.shape))

# sectors for combining output images (clockwise from top)
row = np.arange(-R, R + 1)[:, None]
col = np.arange(-R, R + 1)
sector = np.asarray(ntrans * (1 - np.arctan2(col, row) / np.pi) / 2, dtype=int)

# apply each method --------------------

IM = np.zeros_like(fIM)  # for inverse Abel transformed images
ymax = 0  # max. speed distribution

for i, method in enumerate(transforms):
    print("\n------- {:s} inverse ...".format(method))
    t0 = time()

    # inverse Abel transform using 'method'
    recon = abel.Transform(fIM, method=method, direction="inverse",
                           symmetry_axis=(0, 1)).transform

    print("                    {:.4f} s".format(time()-t0))

    # copy sector to combined output image
    idx = sector == i
    IM[idx] = recon[idx]

    # method label for each quadrant
    annot_angle = 2 * np.pi * (0.5 + i) / ntrans
    annot_coord = (R + 0.8 * R * np.sin(annot_angle),
                   R - 0.8 * R * np.cos(annot_angle))
    ax1.annotate(method, annot_coord, color="white", ha="center")

    # polar projection and speed profile
    radial, speed = abel.tools.vmi.angular_integration_3D(recon)

    # plot speed distribution
    ax2.plot(radial, speed, label=method)

    # update limit
    ymax = max(ymax, speed.max())

plt.suptitle('Dribinski sample image')

ax1.set_title('Inverse Abel comparison')
vmax = IM[:, R+2:].max()  # ignoring pixels near center line
ax1.imshow(IM, vmin=0, vmax=0.1 * vmax)

ax2.set_title('Angular integration')
ax2.set_xlabel('Radial coordinate (pixel)')
ax2.set_xlim(0, 150)
ax2.set_ylabel('Integrated intensity')
ax2.set_ylim(-0.1 * ymax, 1.2 * ymax)
ax2.set_yticks([])
ax2.legend(ncol=2, labelspacing=0.1, frameon=False)

plt.tight_layout()
# plt.savefig('plot_example_all_dribinski.png', dpi=100)
plt.show()
