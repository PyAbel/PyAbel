from BASEX import BASEX
import matplotlib.pyplot as plt

filename = 'data/Xenon_800_nm.tif'
raw_data = plt.imread(filename)

# Specify the center in x,y (horiz,vert) format
center = (681, 491)

print('Performing the inverse Abel transform:')

# Load (or pre-calculate if needed) the basis set for a 1001x1001 image
# using 500 basis function

inv_ab = BASEX(n=1001, nbf=500, basis_dir='./',
        # use_basis_set="../BASEX/data/ascii/original_basis1000{}_1.txt.gz",
        verbose=True, calc_speeds=True)

# Calculate the inverse abel transform for the centered data
recon, speed = inv_ab(raw_data, center, median_size=2,
                    gaussian_blur=0, post_median=0)

#This makes the plots
plt.imshow(recon,origin='lower')
plt.show()
