from abel.basex import BASEX
import matplotlib.pyplot as plt

filename = 'data/Xenon_800_nm.tif'
raw_data = plt.imread(filename)

# Specify the center in x,y (horiz,vert) format
center = (681, 491)

print('Performing the inverse Abel transform:')

# Load (or pre-calculate if needed) the basis set for a 1001x1001 image
# using 500 basis function
# Calculate the inverse abel transform for the centered data
recon, speed = BASEX(raw_data, center, n=1001, basis_dir='./', verbose=True, calc_speeds=True)

#This makes the plots
plt.imshow(recon,origin='lower')
plt.show()
