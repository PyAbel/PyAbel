import numpy as np
import scipy
import matplotlib.pyplot as plt
from abel.tools.analytical import TransformPair
from abel.direct import direct_transform_new

ref = TransformPair(100, profile=6)
noise = 0.05  # RMS intensity of additive noise
tol = 0.04    # RMS tolerance for derivative smoothing

noisy_abel = np.random.RandomState(0).normal(ref.abel, noise)

plt.figure(figsize=(8, 4))

plt.subplot(121)
plt.title('Input function')
plt.xlabel('Radius')
plt.ylabel('Intensity')
plt.plot(ref.r, ref.abel, ':', label='analytical')
plt.plot(ref.r, noisy_abel, label='with noise')
plt.xlim(0, 1)
plt.legend()

# inverse transform using default differentiation and integration
default = direct_transform_new(noisy_abel, r=ref.r, direction='inverse')

# example of a custom derivative function
def derivative(f, x):
    out = np.empty_like(f)
    for row in range(f.shape[0]):
        # create a smoothing spline
        smoothed = scipy.interpolate.make_splrep(x, f[row], s=tol**2 * f.size)
        # sample its first derivative on the original grid
        out[row] = smoothed(x, 1)
    return out

# example of a custom integral function
def integral(f, x):
    return scipy.integrate.simpson(f, x, axis=1)

# inverse transform using custom derivative and integral
custom = direct_transform_new(noisy_abel, r=ref.r, direction='inverse',
                              derivative=derivative, integral=integral,
                              backend='Python')

plt.subplot(122)
plt.title('Inverse Abel transform')
plt.xlabel('Radius')
plt.plot(ref.r, ref.func, ':', label='analytical')
plt.plot(ref.r, default, label='default')
plt.plot(ref.r, custom, label='custom')
plt.xlim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()
