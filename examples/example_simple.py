import abel
original     = abel.tools.analytical.sample_image()
forward_abel = abel.Transform(original, direction='forward', 
                              method='hansenlaw'  ).transform
inverse_abel = abel.Transform(forward_abel, direction='inverse',
                              method='three_point').transform


# plot the original and transform
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(1, 2, figsize=(6, 4))
axs[0].imshow(forward_abel, clim=(0, np.max(forward_abel)*0.6), origin='lower', extent=(-1,1,-1,1))
axs[1].imshow(inverse_abel, clim=(0, np.max(inverse_abel)*0.4), origin='lower', extent=(-1,1,-1,1))

axs[0].set_title('Forward Abel Transform')
axs[1].set_title('Inverse Abel Transform')

plt.tight_layout()
plt.savefig('example.png', dpi=150)
plt.show()
