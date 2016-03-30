# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pyplot as plt

# Dribinski sample image
IM = abel.tools.analytical.sample_image(n=501) 

# split into quadrants
origQ = abel.tools.symmetry.get_image_quadrants(IM)

# speed distribution
orig_speed = abel.tools.vmi.angular_integration(origQ[0], origin=(0,0))

# forward Abel projection
fIM = abel.Transform(IM, direction="forward", method="hansenlaw").transform

# split projected image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(fIM)
Q0 = Q[0].copy()

# onion_bordas inverse Abel transform
borQ0 = abel.onion_bordas.onion_bordas_transform(Q0)
# speed distribution
bor_speed = abel.tools.vmi.angular_integration(borQ0, origin=(0,0))

plt.plot(*orig_speed, linestyle='dashed', label="Dribinski sample")
plt.plot(bor_speed[0], bor_speed[1], label="onion_bordas")
plt.axis(ymin=-0.1)
plt.legend(loc=0)
plt.savefig("example_onion_bordas.png",dpi=100)
plt.show()
