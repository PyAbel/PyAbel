# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import abel.onion_peeling  # had to load directly
import matplotlib.pyplot as plt

# NB this image is even and so not correctly centered
IM = np.loadtxt("data/O2-ANU1024.txt.bz2")

Q = abel.tools.symmetry.get_image_quadrants(IM)
print(Q[0].shape)
Q0 = Q[0].copy()

iOnionQ0 = abel.onion_peeling.onion_peeling_transform(Q[0])
Qradial, Qspeed = abel.tools.vmi.angular_integration(iOnionQ0,origin=(0,0))

iHLQ0 = abel.hansenlaw.hansenlaw_transform(Q0)
HLradial, HLspeed = abel.tools.vmi.angular_integration(iHLQ0,origin=(0,0))

plt.plot(HLradial, HLspeed/HLspeed[50:].max(), label="hansen-law")
plt.plot(Qradial, Qspeed/Qspeed[50:].max(), label="onion_peeling")
plt.title("Full quadrant speed distributions, normalized")
plt.axis(xmin=50, xmax=450, ymin=-0.2, ymax=1.2)
plt.legend(loc=0)
plt.savefig("HL-Onion.png",dpi=100)
plt.show()
