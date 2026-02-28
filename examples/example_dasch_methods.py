import numpy as np
import pyabel
import matplotlib.pyplot as plt

# Dribinski sample image size 501x501
n = 501
IM = pyabel.tools.analytical.SampleImage(n).func

# split into quadrants
origQ = pyabel.tools.symmetry.get_image_quadrants(IM)

# speed distribution of original image
orig_speed = pyabel.tools.vmi.angular_integration_3D(origQ[0], origin=(-1, 0))
scale_factor = orig_speed[1].max()

plt.plot(orig_speed[0], orig_speed[1]/scale_factor, linestyle='dashed',
         label="Dribinski sample")


# forward Abel projection
fIM = pyabel.Transform(IM, direction="forward", method="hansenlaw").transform

# split projected image into quadrants
Q = pyabel.tools.symmetry.get_image_quadrants(fIM)

dasch_transform = {
    "two_point": pyabel.dasch.two_point_transform,
    "three_point": pyabel.dasch.three_point_transform,
    "onion_peeling": pyabel.dasch.onion_peeling_transform
}

for method in dasch_transform.keys():
    Q0 = Q[0].copy()
# method inverse Abel transform
    AQ0 = dasch_transform[method](Q0)
# speed distribution
    speed = pyabel.tools.vmi.angular_integration_3D(AQ0, origin=(-1, 0))

    plt.plot(speed[0], speed[1]*orig_speed[1][14]/speed[1][14]/scale_factor,
             label=method)

plt.title(f'Dasch methods for Dribinski sample image ${n=}$')
plt.xlim((0, 250))
plt.legend(loc='upper center', bbox_to_anchor=(0.35, 1), frameon=False)
plt.tight_layout()
# plt.savefig("plot_example_dasch_methods.png",dpi=100)
plt.show()
