# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel

import matplotlib.pylab as plt

# This example demonstrates ``linbasex`` inverse Abel transform
# of an image obtained using a velocity map imaging (VMI) photoelecton 
# spectrometer to record the photoelectron angular distribution resulting 
# from photodetachement of O2- at 454 nm. 
# Measured at  The Australian National University
# J. Chem. Phys. 133, 174311 (2010) DOI: 10.1063/1.3493349

# Load image as a numpy array - numpy handles .gz, .bz2 
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")
# use scipy.misc.imread(filename) to load image formats (.png, .jpg, etc)

rows, cols = IM.shape    # image size

# Image center should be mid-pixel and image square, 
# `center=convolution` takes care of this

un = [0, 2]  # spherical harmonic orders
proj_angles = range(0, 180, 45)  # projection angles
sig_s = 1  # smoothing Gaussian 1/e width
inc = 1
threshold = 0.2
clip = 0
# Hansen & Law inverse Abel transform
LIM = abel.Transform(IM, method="linbasex", center="convolution",
                     center_options=dict(square=True),
                     transform_options=dict(basis_dir=None, return_Beta=True,
                                            un=un, proj_angles=proj_angles, sig_s=sig_s,
                                            inc=inc, clip=clip,
                                            threshold=threshold)) 

# angular_integration - direct from `linbasex` transform
radial = LIM.radial
speed  = LIM.Beta[0]

# normalize to max intensity peak
speed /= speed[200:].max()  # exclude transform noise near centerline of image

# PAD - photoelectron angular distribution  - direct from `linbasex` transform
beta = LIM.Beta[1]

# plots of the analysis
fig = plt.figure(figsize=(16, 4))
ax1 = plt.subplot2grid((1, 3), (0, 0))
ax2 = plt.subplot2grid((1, 3), (0, 1))
ax3 = plt.subplot2grid((1, 3), (0, 2), sharex=ax2)

# join 1/2 raw data : 1/2 inversion image
inv_IM = LIM.transform
cols = inv_IM.shape[1]
c2 = cols//2 + cols % 2
vmax = IM[:, :c2-100].max()
inv_IM *= vmax/inv_IM[:, c2+100:].max()
JIM = np.concatenate((IM[:, :c2], inv_IM[:, c2:]), axis=1)

# Prettify the plot a little bit:
# Plot the raw data
im1 = ax1.imshow(JIM, origin='lower', aspect='auto', vmin=0, vmax=vmax)
#fig.colorbar(im1, ax=ax1, fraction=.1, shrink=0.9, pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('VMI, inverse Abel: {:d}x{:d}'.format(*inv_IM.shape), fontsize='small')

# Plot the 1D speed distribution
ax2.plot(radial, speed)
ax2.set_xlabel('radial pixel')
ax2.axis(ymin=-0.1, ymax=1.2)
ax2.set_ylabel('intensity')
ax2.set_title('Beta[0]: speed distribution', fontsize='small')
ax2.set_xlabel('radial coordinate (pixels)')

# Plot anisotropy variation
ax3.plot(radial, beta, 'r-')
ax3.axis(xmin=150, xmax=450, ymin=-1.2, ymax=0.1)
ax3.set_ylabel("$\\beta$")
ax3.set_title("Beta[1]: anisotropy parameter", fontsize='small')
ax2.set_xlabel('radial coordinate (pixels)')

plt.suptitle("un={}, an={}, sig_s={}".format(un, proj_angles, sig_s))

# Save a image of the plot
plt.savefig("example_linbasex.png", dpi=100)

# Show the plots
plt.show()
