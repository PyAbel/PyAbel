#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
from abel.hansenlaw import *
from abel.tools import *
import matplotlib.pylab as plt

def P2(x):   # 2nd order Legendre polynomial
    return (3*x*x-1)/2

def PAD(theta, beta, amp):
    return amp*(1 + beta*P2(np.cos(theta))) 

# experimental data, O2- photodetachment velocity-map image
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")

# inverse Abel transform
AIM = iabel_hansenlaw(IM)

# radial ranges (of spectral features) for which to follow intensity vs angle
r_range=[(93,111),(145,162),(255,280),(330,350),(350,370),(370,390),(390,410),(410,430)]

# map to intensity vs theta for each radial range
theta, intensities = calculate_angular_distributions(AIM, r_range)

print ("radial-range      anisotropy parameter (β)")
for rr,intensity in zip(r_range,intensities):
    beta, amp  = anisotropy_parameter(theta, intensity)
    result = "    {:3d}-{:3d}        {:+.2f}+-{:.2f}".format(*rr+beta)
    print (result)

# plot one example radial intensity variation and fit
rr = r_range[3]
intensity = intensities[3]
beta, amp  = anisotropy_parameter(theta, intensity)
plt.subplot(121)
rows,cols = AIM.shape
rw2 = rows//2
cl2 = rows//2
vmax = AIM[rw2+50:,:].max() # max image intensity, exclude center line 
# draw a circle representing this radial range
for rw in range(rows):
   for cl in range(cols):
       circ = (rw-rw2)**2 + (cl-cl2)**2
       if circ >= rr[0]**2 and circ <= rr[1]**2:
           AIM[rw,cl] = vmax

plt.title("intensity along red circle")
plt.imshow(AIM,vmin=0,vmax=vmax*0.5)
plt.subplot(122)
plt.plot (theta, intensity, 'r',
          label="expt. data r=[{:d}:{:d}]".format(*rr))
plt.plot (theta, PAD(theta, beta[0], amp[0]), 'b', lw=2, label="fit")
plt.annotate("$\\beta = ${:+.2f}+-{:.2f}".format(*beta),(-2,-1.1))
plt.legend(loc=1,labelspacing=0.1,fontsize='small')

plt.axis(ymin=-2,ymax=9)
plt.xlabel("angle $\\theta$ (radians)")
plt.ylabel("intensity")

#plt.title("$I \propto 1 + \\beta P_{2}(\cos\\theta)$")
plt.suptitle("example_anisotropy_parameter.py",fontsize="large")
plt.savefig("example_anisotropy_parameter.png",dpi=150)
plt.show()
