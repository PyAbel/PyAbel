import abel
original     = abel.tools.analytical.sample_image()
forward_abel = abel.transform(original,     direction='forward', method='hansenlaw'  )['transform']
inverse_abel = abel.transform(forward_abel, direction='inverse', method='three_point')['transform']


# plot the original and transform
import matplotlib.pyplot as plt
import numpy as np
fig, axs = plt.subplots(1,2,figsize=(7,5))
axs[0].imshow(forward_abel,clim=(0,np.max(forward_abel)*0.3))
axs[1].imshow(inverse_abel,clim=(0,np.max(inverse_abel)*0.3))

axs[0].set_title('Forward Abel Transform')
axs[1].set_title('Inverse Abel Transform')

plt.show()