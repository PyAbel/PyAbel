# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel

import matplotlib.pylab as plt

# Load image as a numpy array - numpy handles .gz, .bz2 
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")
# use scipy.misc.imread(filename) to load image formats (.png, .jpg, etc)

# Hansen & Law inverse Abel transform
AIM = abel.Transform(IM, center="slice", method="hansenlaw",
                     symmetry_axis=None).transform

# PES - photoelectron speed distribution  -------------
print('Calculating speed distribution:')

r, speed  = abel.tools.vmi.angular_integration(AIM)

# normalize to max intensity peak
speed /= speed[200:].max()  # exclude transform noise near centerline of image

# PAD - photoelectron angular distribution  ------------
# radial ranges (of spectral features) to follow intensity vs angle
# view the speed distribution to determine radial ranges
r_range = [(93, 111), (145, 162), (255, 280), (330, 350), (350, 370), 
           (370, 390), (390, 410), (410, 430)]

# anisotropy parameter for each tuple r_range
Beta, Amp, R = abel.tools.vmi.anisotropy(AIM, r_range)

# OR  anisotropy parameter for ranges (0, 20), (20, 40) ...
Beta_whole_grid, Amp_whole_grid, Radial_midpoints =\
                         abel.tools.vmi.anisotropy(AIM, 20)

# plots of the analysis
fig = plt.figure(figsize=(8, 4))
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# join 1/2 raw data : 1/2 inversion image
rows, cols = IM.shape
c2 = cols//2
vmax = IM[:, :c2-100].max()
AIM *= vmax/AIM[:, c2+100:].max()
JIM = np.concatenate((IM[:, :c2], AIM[:, c2:]), axis=1)

# Prettify the plot a little bit:
# Plot the raw data
im1 = ax1.imshow(JIM, origin='lower', aspect='auto', vmin=0, vmax=vmax)
fig.colorbar(im1, ax=ax1, fraction=.1, shrink=0.9, pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('VMI, inverse Abel: {:d}x{:d}'\
              .format(rows, cols))

# Plot the 1D speed distribution
ax2.plot(speed, label='speed')
BT = np.transpose(Beta)
ax2.errorbar(R, BT[0], BT[1], fmt='o', color='r', label='specific radii')
BrT = np.transpose(Beta_whole_grid)
ax2.plot(Radial_midpoints, BrT[0], '-g', label='stepped')
ax2.axis(xmax=450, ymin=-1.2, ymax=1.2)
ax2.set_xlabel('radial pixel')
ax2.set_ylabel('speed/anisotropy')
ax2.set_title('speed/anisotropy distribution')
ax2.legend(frameon=False, labelspacing=0.1, numpoints=1, loc=3, fontsize='small')

plt.subplots_adjust(left=0.06, bottom=0.17, right=0.95, top=0.89, 
                    wspace=0.35, hspace=0.37)

# Save a image of the plot
plt.savefig("example_PAD.png", dpi=100)

# Show the plots
plt.show()
