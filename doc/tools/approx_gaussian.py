import numpy as np
from matplotlib import pyplot as plt
import pyabel

from pyabel.tools.polynomial import ApproxGaussian, PiecewisePolynomial

r = np.arange(201)
r0 = 100
sigma = 20
# actual Gaussian function
gauss = np.exp(-((r - r0) / sigma)**2 / 2)
# approximation with default tolerance (~0.5%)
ranges = ApproxGaussian().scaled(1, r0, sigma)
approx = PiecewisePolynomial(r, ranges)
nodes = [rng[0] for rng in ranges] + [ranges[-1][1]]
# much more accurate approximation as a reference
ref = PiecewisePolynomial(r, ApproxGaussian(1e-5).scaled(1, r0, sigma))


def plot(title, f, g, lim):
    plt.figure(figsize=(6, 4), frameon=False, dpi=200)

    plt.subplot(3, 1, (1, 2))
    plt.title(title)
    for R in nodes:
        plt.axvline(R, lw=0.5, c='lightgray')
    plt.axhline(lw=0.5, c='k')
    plt.plot(r, g, label='Gaussian')
    plt.plot(r, f, '--', label='approx.')
    plt.autoscale(axis='x', tight=True)
    plt.ylabel('Value')
    plt.legend()

    plt.subplot(3, 1, 3)
    for R in nodes:
        plt.axvline(R, lw=0.5, c='lightgray')
    plt.axhline(lw=0.5, c='k')
    plt.plot(r, f - g, c='brown')
    plt.autoscale(axis='x', tight=True)
    plt.xlabel('Radius, px')
    plt.ylim(-lim, lim)
    plt.ylabel('Difference')

    plt.tight_layout(pad=0)
    plt.show()


if __name__ == '__main__':
    M = ref.func.max()
    D = (approx.func - gauss).max()
    print('max func =', M)
    print('max diff =', D)
    print('rel diff =', D / M * 100, '%')
    plot('func', approx.func, gauss, 0.005)

    M = ref.pyabel.max()
    D = (approx.abel - ref.abel).max()
    print('max abel =', M)
    print('max diff =', D)
    print('rel diff =', D / M * 100, '%')
    plot('abel', approx.abel, ref.abel, 0.35)
