# -*- coding: utf-8 -*-
import numpy as np
import abel
import matplotlib.pyplot as plt

IM = np.loadtxt("data/VMI_art1.txt.bz2")

un = [0, 2, 4]  # Legendre polynomial orders
an = range(0, 180, 10)  # projection angles in 10 degree steps
inc = 1  # pixel grid
sig_s = 1  # smoothing Gaussian convolution
threshold = 0.2  # threshold for normalization of higher order Newton spheres
clip=0  # clip first vectors (smallest Newton spheres) to avoid singularities

# linbasex method - center ensures image has odd square shape
#                 - speed and anisotropy parameters evaluated by method
LIM = abel.Transform(IM, method='linbasex', center='convolution',
                     center_options=dict(square=True),
                     transform_options=dict(basis_dir='./', an=an, inc=inc,
                              sig_s=sig_s, threshold=threshold, clip=clip, 
                              return_Beta=True, verbose=True))

# hansenlaw method - speed and anisotropy parameters evaluated by integration
HIM = abel.Transform(IM, method="hansenlaw", center='convolution', 
                     center_options=dict(square=True),
                     angular_integration=True)

# alternative derivation of anisotropy parameters via integration
rrange = [(20, 50), (60, 80), (85, 100), (125, 155), (185, 205), (220, 240)]
intensity, theta, rr  = abel.tools.vmi.radial_integration(HIM.transform,
                   radial_ranges=rrange)

# anisotropy parameter from integrated intensity (from hansenlaw method)
beta = []
ebeta = []
rr = []
for i, inten in enumerate(intensity):
    betax, amp = abel.tools.vmi.anisotropy_parameter(theta, inten)
    ar = np.average(rrange[i])
    beta.append(betax[0])
    ebeta.append(betax[1])
    rr.append(ar)

plt.figure(figsize=(12, 6))
ax0 = plt.subplot2grid((2,4), (0,0))
ax3 = plt.subplot2grid((2,4), (1,0))
ax1 = plt.subplot2grid((2,4), (0,1), colspan=2, rowspan=2)
ax2 = plt.subplot2grid((2,4), (0,3), sharex=ax1, rowspan=2)

ax0.imshow(LIM.transform, vmin=0, vmax=LIM.transform.max()*2/3)
ax0.set_aspect('equal')
ax0.axis('off')
ax0.invert_yaxis()
ax0.set_title("linbasex")
ax3.imshow(HIM.transform, vmin=0, vmax=HIM.transform[200:].max()*1/5)
ax3.axis('off')
#ax3.axis(xmin=750, xmax=850, ymin=420, ymax=620)
ax3.invert_yaxis()
ax3.set_aspect('equal')
ax3.set_title("hansenlaw")

ax1.plot(LIM.radial, LIM.Beta[0], 'r-', label='linbasex')
ax1.plot(HIM.angular_integration[1]/HIM.angular_integration[1].max(),
         'b-', label='hansenlaw')
ax1.legend(loc=0, labelspacing=0.1, frameon=False, numpoints=1, fontsize=10)
ax1.set_title("Beta0 norm an={} un={} inc={} sig={} th={}".
              format(an, un, inc, sig_s, threshold), fontsize=10)
ax1.axis(ymin=-0.1, ymax=1.2)
ax1.set_xlabel("radial coordinate (pixels)")

ax2.plot(LIM.radial, LIM.Beta[1], 'r-', label='linbasex')
ax2.errorbar(x=rr, y=beta, yerr=ebeta, color='b', lw=2, fmt='o',
             label='hansenlaw')
ax2.set_title(r"$\beta$-parameter  (Beta2 norm)", fontsize=10)
ax2.legend(loc=0, labelspacing=0.1, frameon=False, numpoints=1, fontsize=10)
ax2.axis(xmax=300, ymin=-1.0, ymax=1.0)
ax2.set_xlabel("radial coordinate (pixels)")

plt.savefig("example_linbasex_hansenlaw.png", dpi=100)
plt.show()
