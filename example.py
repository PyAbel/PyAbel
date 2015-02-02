import BASEX
import matplotlib.pyplot as plt

filename = 'example_data/Xenon_800_nm.tif'
raw_data = plt.imread(filename)

# Specify the center in x,y (horiz,vert) format
center = (681,491)

print 'Performing the inverse Abel transform:'
# Transform the data
recon,speeds = BASEX.center_and_transform(raw_data,center,median_size=2,gaussian_blur=0,
                                    post_median=0,verbose=True)
									
plt.imshow(recon,origin='lower')
plt.show()