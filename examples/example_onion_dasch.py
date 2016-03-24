# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import abel
import matplotlib.pyplot as plt

IM = abel.tools.analytical.sample_image(n=501)  # Dribinski sample image

# forward Abel projection
fIM = abel.Transform(IM, direction="forward", method="hansenlaw").transform

# split into quadrants
origQ = abel.tools.symmetry.get_image_quadrants(IM)

# speed distribution
orig_speed = abel.tools.vmi.angular_integration(origQ[0], origin=(0,0))
scale_factor = orig_speed[1].max()

# split projected image into quadrants
Q = abel.tools.symmetry.get_image_quadrants(fIM)
Q0 = Q[0].copy()

# dasch_onion_peeling inverse Abel transform
dopQ0 = abel.onion_dasch.onion_dasch_transform(Q0)
# speed distribution
dop_speed = abel.tools.vmi.angular_integration(dopQ0, origin=(0,0))

plt.plot(orig_speed[0], orig_speed[1]/scale_factor, linestyle='dashed', label="Dribinski sample")
plt.plot(dop_speed[0], dop_speed[1]*orig_speed[1][14]/dop_speed[1][14]/scale_factor, label="onion_dasch")
plt.axis(ymin=-0.1)
plt.legend(loc=0)
plt.savefig("example_onion_dasch.png",dpi=100)
plt.show()
