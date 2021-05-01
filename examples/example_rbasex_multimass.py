from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt

import abel
from abel.rbasex import rbasex_transform

# This example demonstrates analysis of partially overlapping velocity-map
# images, which might be useful for the "multimass imaging" method (see
# https://pubs.acs.org/doi/10.1021/jp053143m).
# Reconstruction of each part of the combined image uses zero weights for all
# pixels not belonging to that part or contaminated by signals from other
# parts in order to extract distributions pertaining to that part only.
# Notice that the central image in this example cannot be correctly
# reconstructed by "usual" Abel-transform methods, since all its quadrants are
# contaminated. However, the remaining uncontaminated radial and angular ranges
# are sufficient for rBasex to reconstruct the complete distributions.

# Create an artificial sample image from experimental and synthetic images.
# central image
im2 = np.loadtxt('data/O2-ANU1024.txt.bz2')
from scipy.ndimage import zoom
im2 = zoom(im2, 0.75)  # (resized for better visibility)
h2, w2 = im2.shape
# right image
im3 = np.loadtxt('data/VMI_art1.txt.bz2')
im3 *= 2  # (intensified for better visibility)
h3, w3 = im3.shape
# left image
h1 = w1 = min(h3, w3)
r1 = h1 // 2
x = np.linspace(-r1, r1, w1)
im1 = np.random.RandomState(0).poisson(200 * np.exp(-(x**2 + x[:, None]**2) /
                                       (r1 / 3)**2))
# assemble the whole image
h, w = max(h2, h3), w2 + w3
im = np.zeros((h, w))
row1, col1 = (h - h1) // 2, 0
im[row1:row1+h1, col1:col1+w1] += im1
row2, col2 = (h - h2) // 2, (w - w2) // 2
im[row2:row2+h2, col2:col2+w2] += im2
row3, col3 = (h - h3) // 2, w - w3
im[row3:row3+h3, col3:col3+w3] += im3

# Origins and maximal radii for each part
# (in reality they will need to be determined from the data somehow; also,
# in practice it would be better to cut the whole image into parts and work
# with them separately, which is not done here to simplify the code).
# for left image
origin1 = (h // 2 - 1, w1 // 2)
r1 = min(h1, w1) // 2
# for central image
origin2 = (h // 2, w // 2)
r2 = min(h2, w2) // 2 - 50
# for right image
origin3 = (h // 2 - 1, w - w3 // 2 - 1)
r3 = min(h3, w3) // 2

# Create "masks" for each part with unit weights for "good" pixels and zero
# weights for "bad" pixels.
# coordinates relative to each origin
x1, y1 = abel.tools.polar.index_coords(im, origin=origin1)
x2, y2 = abel.tools.polar.index_coords(im, origin=origin2)
x3, y3 = abel.tools.polar.index_coords(im, origin=origin3)
# for left image (include left, exclude central)
mask1 = np.array((x1**2 + y1**2 < r1**2) *  # inside radius r1 from origin1 and
                 (x2**2 + y2**2 > r2**2),   # outside radius r2 from origin2
                 dtype=float)
# for central image (include central, exclude left and right)
mask2 = np.array((x2**2 + y2**2 < r2**2) *  # inside radius r2 from origin2 and
                 (x1**2 + y1**2 > r1**2) *  # outside radius r1 from origin1 and
                 (x3**2 + y3**2 > r3**2),   # outside radius r3 from origin3
                 dtype=float)
# for right image (include right, exclude central)
mask3 = np.array((x3**2 + y3**2 < r3**2) *  # inside radius r3 from origin3 and
                 (x2**2 + y2**2 > r2**2),   # outside radius r2 from origin2
                 dtype=float)

fig = plt.figure(figsize=(12, 8))

# Show the whole image
plt.subplot(221)
plt.title('Partially overlapping images\n'
          '(with outlined regions for analysis)')
plt.imshow(im, cmap='hot')
# overlay with the boundaries of each mask (only for demonstration)
from scipy.ndimage import binary_erosion
brush = np.ones((11, 11))
dmask = 1 * (mask1 - binary_erosion(mask1, structure=brush)) + \
        2 * (mask2 - binary_erosion(mask2, structure=brush)) + \
        3 * (mask3 - binary_erosion(mask3, structure=brush))
dmask[dmask == 0] = np.nan
from matplotlib.colors import ListedColormap
rgb = ListedColormap(['#CC0000', '#00AA00', '#0055FF'])
plt.imshow(dmask, extent=(0, w, 0, h), cmap=rgb, interpolation='nearest')

# Analyze the left part and plot results
plt.subplot(222)
plt.title('Distributions: left image')
# the reconstructed image is not used in this example, so it is not created;
# also notice that order=0 is enough for this totally isotropic case
_, distr = rbasex_transform(im, origin=origin1, rmax=r1, order=0, weights=mask1, out=None)
r, I = distr.rIbeta()
plt.plot(r, I, c=rgb(0), label='$I(r)$')
plt.legend()
plt.autoscale(enable=True, tight=True)

# Analyze the central part and plot results
plt.subplot(223)
plt.title('Distributions: central image')
# here the default order=2 is needed and used
_, distr = rbasex_transform(im, origin=origin2, rmax=r2, weights=mask2, out=None)
r, I, beta = distr.rIbeta()
plt.plot(r, I, c=rgb(1), label='$I(r)$')
# beta(r) I(r) is the "speed distribution" of P_2(r)
plt.plot(r, beta * I, c='gray', label='$\\beta(r) \\cdot I(r)$')
plt.legend()
plt.autoscale(enable=True, tight=True)

# Analyze the right part and plot results
plt.subplot(224)
plt.title('Distributions: right image')
_, distr = rbasex_transform(im, origin=origin3, rmax=r3, weights=mask3, out=None)
r, I, beta = distr.rIbeta()
plt.plot(r, I, c=rgb(2), label='$I(r)$')
plt.plot(r, I * beta, c='gray', label='$\\beta(r) \\cdot I(r)$')
plt.legend()
plt.autoscale(enable=True, tight=True)

plt.tight_layout()
plt.show()
