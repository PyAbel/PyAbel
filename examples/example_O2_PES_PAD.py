# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel

import matplotlib.pylab as plt

# This example demonstrates Hansen and Law inverse Abel transform
# of an image obtained using a velocity map imaging (VMI) photoelecton 
# spectrometer to record the photoelectron angular distribution resulting 
# from photodetachement of O2- at 454 nm. 
# Measured at  The Australian National University
# J. Chem. Phys. 133, 174311 (2010) DOI: 10.1063/1.3493349

# Load image as a numpy array - numpy handles .gz, .bz2 
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")
# use scipy.misc.imread(filename) to load image formats (.png, .jpg, etc)

rows, cols = IM.shape    # image size

# Image center should be mid-pixel, i.e. odd number of colums
if cols % 2 != 1: 
    print ("even pixel width image, make it odd and re-adjust image center")
    IM = abel.tools.center.center_image(IM, center="slice")
    rows, cols = IM.shape   # new image size

r2 = rows//2   # half-height image size
c2 = cols//2   # half-width image size

# Hansen & Law inverse Abel transform
AIM = abel.Transform(IM, method="hansenlaw", direction="inverse",
                     symmetry_axis=None).transform

# PES - photoelectron speed distribution  -------------
print('Calculating speed distribution:')

r, speed  = abel.tools.vmi.angular_integration(AIM)

# normalize to max intensity peak
speed /= speed[200:].max()  # exclude transform noise near centerline of image

# PAD - photoelectron angular distribution  ------------
print('Calculating angular distribution:')
# radial ranges (of spectral features) to follow intensity vs angle
# view the speed distribution to determine radial ranges
r_range = [(93, 111), (145, 162), (255, 280), (330, 350), (350, 370), 
           (370, 390), (390, 410), (410, 430)]

# map to intensity vs theta for each radial range
Beta, Amp, rad,intensities, theta = abel.tools.vmi.radial_integration(AIM, radial_ranges=r_range)

print("radial-range      anisotropy parameter (beta)")
for beta, rr in zip(Beta, r_range):
    result = "    {:3d}-{:3d}        {:+.2f}+-{:.2f}"\
             .format(rr[0], rr[1], beta[0], beta[1])
    print(result)

# plots of the analysis
fig = plt.figure(figsize=(15, 4))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)

# join 1/2 raw data : 1/2 inversion image
vmax = IM[:, :c2-100].max()
AIM *= vmax/AIM[:, c2+100:].max()
JIM = np.concatenate((IM[:, :c2], AIM[:, c2:]), axis=1)
rr = r_range[-3]
intensity = intensities[-3]
beta, amp = Beta[-3], Amp[-3]

# Prettify the plot a little bit:
# Plot the raw data
im1 = ax1.imshow(JIM, origin='lower', aspect='auto', vmin=0, vmax=vmax)
fig.colorbar(im1, ax=ax1, fraction=.1, shrink=0.9, pad=0.03)
ax1.set_xlabel('x (pixels)')
ax1.set_ylabel('y (pixels)')
ax1.set_title('VMI, inverse Abel: {:d}x{:d}'\
              .format(rows, cols))

# Plot the 1D speed distribution
ax2.plot(speed)
ax2.plot((rr[0], rr[0], rr[1], rr[1]), (1, 1.1, 1.1, 1), 'r-')  # red highlight
ax2.axis(xmax=450, ymin=-0.05, ymax=1.2)
ax2.set_xlabel('radial pixel')
ax2.set_ylabel('intensity')
ax2.set_title('speed distribution')

# Plot anisotropy variation
ax3.plot(theta, intensity, 'r',
         label="expt. data r=[{:d}:{:d}]".format(*rr))


def P2(x):   # 2nd order Legendre polynomial
    return (3*x*x-1)/2


def PAD(theta, beta, amp):
    return amp*(1 + beta*P2(np.cos(theta)))


ax3.plot(theta, PAD(theta, beta[0], amp[0]), 'b', lw=2, label="fit")
ax3.annotate("$\\beta = ${:+.2f}+-{:.2f}".format(*beta), (-2, -1.1))
ax3.legend(loc=1, labelspacing=0.1, fontsize='small')

ax3.axis(ymin=-2, ymax=12)
ax3.set_xlabel("angle $\\theta$ (radians)")
ax3.set_ylabel("intensity")
ax3.set_title("anisotropy parameter")


# Plot the angular distribution 
plt.subplots_adjust(left=0.06, bottom=0.17, right=0.95, top=0.89, 
                    wspace=0.35, hspace=0.37)

# Save a image of the plot
plt.savefig("example_O2_PES_PAD.png", dpi=100)

# Show the plots
plt.show()
