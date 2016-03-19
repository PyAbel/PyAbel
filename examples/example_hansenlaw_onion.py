# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pyplot as plt

IM = abel.tools.analytical.sample_image(n=1001)
fIM = abel.Transform(IM, direction="forward", method="hansenlaw").transform

Q = abel.tools.symmetry.get_image_quadrants(fIM)
Q0 = Q[0].copy()

iOnionQ0 = abel.onion_peeling.onion_peeling_transform(Q[0])
Qradial, Qspeed = abel.tools.vmi.angular_integration(iOnionQ0,origin=(0,0))

iHLQ0 = abel.hansenlaw.hansenlaw_transform(Q0)
HLradial, HLspeed = abel.tools.vmi.angular_integration(iHLQ0,origin=(0,0))



plt.plot(HLradial, HLspeed/HLspeed[50:].max(), label="hansen-law")
plt.plot(Qradial, Qspeed/Qspeed[50:].max(), label="onion_peeling")
plt.title("hansen-law and onion-peeling: Dribinski sample image $n=1001$")
plt.axis(ymin=-0.1)
plt.legend(loc=0)
plt.savefig("example_hansenlaw_onion.png",dpi=100)
plt.show()
