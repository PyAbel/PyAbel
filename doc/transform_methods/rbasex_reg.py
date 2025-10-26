import numpy as np
import matplotlib.pyplot as plt

from abel.tools.analytical import SampleImage
from abel.tools.vmi import Ibeta
from abel.rbasex import rbasex_transform

rmax = 200
scale = 10000

vlim = 0.5

Ilim = (-500, 2500)
dIlim = (-700, 700)

def rescaleI(im):
    return np.sqrt(np.abs(im)) * np.sign(im)

# calculate relative RMS error for given regularization parameters
def plot(method, strength=None):
    # test distribution
    source = SampleImage(n=2 * rmax - 1).func / scale
    Isrc, _ = Ibeta(source)
    Inorm = (Isrc**2).sum()

    # simulated projection fith Poissonian noise
    proj, _ = rbasex_transform(source, direction='forward')
    proj[proj < 0] = 0
    proj = np.random.RandomState(0).poisson(proj)  # (reproducible, see NEP 19)

    # reconstructed image and intensity
    if strength is None:
        reg = method
    else:
        reg = (method, strength)
    im, distr = rbasex_transform(proj, reg=reg)
    I, _ = distr.Ibeta()

    # plot...
    fig = plt.figure(figsize=(7, 3.5), frameon=False)

    # image
    plt.subplot(121)

    fig = plt.imshow(rescaleI(im), vmin=-vlim, vmax=vlim, cmap='bwr')

    plt.axis('off')
    plt.text(0, 2 * rmax, str(method), va='top')

    # intensity
    ax = plt.subplot2grid((3, 2), (0, 1), rowspan=2)

    ax.plot(Isrc, c='r', lw=1)
    ax.plot(I, c='k', lw=1)

    ax.set_xlim((0, rmax))
    ax.set_ylim(Ilim)

    # error
    plt.subplot(326)

    plt.axhline(c='r', lw=1)
    plt.plot(I - Isrc, c='b', lw=1)

    plt.xlim((0, rmax))
    plt.ylim(dIlim)

    # finish
    plt.subplots_adjust(left=0, right=0.97, wspace=0.1,
                        bottom=0.08, top=0.98, hspace=0.5)
    #plt.savefig('rbasex_reg.svg')
    #plt.show()

#plot(None)
