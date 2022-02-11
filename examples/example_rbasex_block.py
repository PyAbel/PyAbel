from __future__ import division, print_function

import numpy as np
import matplotlib.pyplot as plt
from copy import copy

import abel
from abel.tools.analytical import SampleImage
from abel.tools.vmi import rharmonics
from abel.rbasex import rbasex_transform

# This example demonstrates analysis of velocity-map images with "damaged"
# areas, in this case, with some parts obstructed by a beam block (see
# https://aip.scitation.org/doi/10.1063/1.4921990 for a practical example).
# First, a general Abel-transform method is used naively to demonstrate
# artifacts produced in the reconstructed image and the extracted radial
# distributions.
# Second, radial distributions are extracted from the same reconstructed image,
# but with its artifacts masked, showing good agreement with the actual
# distributions.
# Third, the rBasex method is used to transform the initial damaged image with
# the experimental artifacts masked, yielding a correct and cleaner
# reconstructed image and correct reconstructed distributions.

R = 150  # image radius
N = 2 * R + 1  # full image width and height
block_r = 20  # beam-block disk radius
block_w = 5  # beam-block holder width

vlim = 3.6  # intensity limits for images
ylim = (-1.3, 2.7)  # limits for plots

# create source distribution and its profiles for reference
source = SampleImage(N).func / 100
r_src, P0_src, P2_src = rharmonics(source)

# simulate experimental image:
# projection
im, _ = rbasex_transform(source, direction='forward')
# Poissonian noise
im[im < 0] = 0
im = np.random.RandomState(0).poisson(im)
# image coordinates
im_x = np.arange(float(N)) - R
im_y = R - np.arange(float(N))[:, None]
im_r = np.sqrt(im_x**2 + im_y**2)
# simulate beam-block shadow
im = im / (1 + np.exp(-(im_r - block_r)))
im[:R] *= 1 / (1 + np.exp(-(np.abs(im_x) - block_w)))

# create mask that fully covers beam-block shadow
mask_r = block_r + 5
mask_w = block_w + 5
mask = np.ones_like(im)
mask[im_r < mask_r] = 0
mask[:R, R-mask_w:R+mask_w] = 0

# reconstruct "as is" by a general Abel-transform method
rec_abel = abel.Transform(im, method='two_point').transform
# extract profiles "as is"
r_abel, P0_abel, P2_abel = rharmonics(rec_abel)
# extract profiles from masked reconstruction
r_abel_masked, P0_abel_masked, P2_abel_masked = rharmonics(rec_abel, weights=mask)

# reconstruct masked image with rBasex
rec_rbasex, distr_rbasex = rbasex_transform(im, weights=mask)
r_rbasex, P0_rbasex, P2_rbasex = distr_rbasex.rharmonics()

# plotting...
plt.figure(figsize=(7, 7))

cmap_hot = copy(plt.cm.hot)
cmap_hot.set_bad('lightgray')
cmap_seismic = copy(plt.cm.seismic)
cmap_seismic.set_bad('lightgray')

def plots(row,
          im_title, im, im_mask,
          tr_title, tr, tr_mask,
          pr_title, r, P0, P2):
    # input image
    if im is not None:
        plt.subplot(3, 4, 4 * row + 1)
        plt.title(im_title)
        im_masked = np.ma.masked_where(im_mask == 0, im)
        plt.imshow(im_masked, cmap=cmap_hot)
        plt.axis('off')

    # transformed image
    plt.subplot(3, 4, 4 * row + 2)
    plt.title(tr_title)
    tr_masked = np.ma.masked_where(tr_mask == 0, tr)
    plt.imshow(tr_masked, vmin=-vlim, vmax=vlim, cmap=cmap_seismic)
    plt.axis('off')

    # profiles
    plt.subplot(3, 2, 2 * row + 2)
    plt.title(pr_title)
    plt.axvspan(0, mask_r, color='lightgray')  # shade region without valid data
    plt.plot(r_src, P0_src, 'C0--', lw=1)
    plt.plot(r_src, P2_src, 'C3--', lw=1)
    plt.plot(r, P0, 'C0', lw=1, label='$P_0(r)$')
    plt.plot(r, P2, 'C3', lw=1, label='$P_2(r)$')
    plt.xlim((0, R))
    plt.ylim(ylim)
    plt.legend()

plots(0,
      'Raw image', im, None,
      'Two-point', rec_abel, None,
      'Profiles', r_abel, P0_abel, P2_abel)

plots(1,
      None, None, None,
      'Two-point + mask', rec_abel, mask,
      'Masked profiles', r_abel_masked, P0_abel_masked, P2_abel_masked)

plots(2,
      'Masked image', im, mask,
      'rBasex', rec_rbasex, None,
      'Profiles', r_rbasex, P0_rbasex, P2_rbasex)

plt.subplots_adjust(left=0.01, right=0.97, wspace=0.1,
                    bottom=0.04, top=0.96, hspace=0.3)
plt.show()
