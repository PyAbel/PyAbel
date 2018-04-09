import numpy as np
import abel
import matplotlib.pyplot as plt


def hansen_transform(im, dr=1, direction='inverse', hold_order=1):
    # Hansen IEEE Trans. Acoust. Speech Signal Proc. 33, 666 (1985)
    #  10.1109/TASSP.1985.1164579

    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    def phi(n, lam):
        return (n/(n-1))**lam

    # state equation integration
    def I(n, lam, a):  # integral (epsilon/r)^(lamda+pwr)
        lama = lam + a
        return (1 - (n/(n-1))**lama)*(n-1)**a/lama

    def I0(n, lam, a):
        # special case divide issue for lamda=0, only for inverse transform
        integral = np.empty_like(lam)
        integral[0] = -np.log(n/(n-1))
        integral[1:] = (1 - phi(n, lam[1:]))/lam[1:]

        return integral

    # first-order hold functions
    def beta0(n, lam, a, In):  # fn   q\epsilon  +  p
        return I(n, lam, a+1) - (n-1)*In(n, lam, a)

    def beta1(n, lam, a, In):  # fn-1   p + q\epsilon
        return n*In(n, lam, a) - I(n, lam, a+1)

    if direction == 'forward':
        drive = im.copy()
        h *= -2*dr*np.pi  # include Jacobian with h-array
        a = 1
        intfunc = I
    else:  # inverse Abel transform
        drive = np.gradient(im, dr)
        a = 0  # from 1/piR factor
        intfunc = I0  # special case for lam=0

    aim = np.zeros_like(im)  # Abel transform array
    cols = im.shape[-1]
    N = np.arange(cols-1, 1, -1)

    x = np.zeros(h.size)

    if hold_order:  # Hansen first-order hold approximation
        for n in N:
            x = phi(n, lam)*x + beta0(n, lam, a, intfunc)*h*drive[n]\
                              + beta1(n, lam, a, intfunc)*h*drive[n-1]
            aim[n-1] = x.sum()

    else:  # Hansen & Law zero-order hold approximation
        for n in N:
            x = phi(n, lam)*x + intfunc(n, lam, a)*h*drive[n-1]
            aim[n-1] = x.sum()

    # missing 1st column
    aim[0] = aim[1]

    return aim

if __name__ == '__main__':

    n = 101
    hold_order = 1

    f = abel.tools.analytical.TransformPair(n, 3)

    hf = hansen_transform(f.func, dr=f.dr, direction='forward',
                          hold_order=hold_order)
    msef = np.square(hf-f.abel).mean()

    hi = hansen_transform(f.abel, dr=f.dr, hold_order=hold_order)
    msei = np.square(hi-f.func).mean()

    # plotting ---
    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.plot(f.r, f.abel, label=f.label)
    ax0.plot(f.r, hf, label='Hansen')
    ax0.legend(fontsize='smaller', labelspacing=0.1)
    ax0.annotate('mse={:4.2e}'.format(msef), (0, 0.05))
    ax0.set_xlabel(r'radius')
    ax0.set_ylabel(r'projection')
    ax0.set_title('forward')

    ax1.plot(f.r, f.func, label=f.label)
    ax1.plot(f.r, hi, label='Hansen')
    ax1.legend(fontsize='smaller', labelspacing=0.1)
    ax1.annotate('mse={:4.2e}'.format(msei), (0, 0.05))
    ax1.set_xlabel(r'radius')
    ax1.set_ylabel(r'source')
    ax1.set_title('inverse')

    plt.suptitle(r'Hansen {:s}-order hold, curve A ({:s}), N={:d}'.
                 format('first' if hold_order else 'zero', f.label, n))
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('hansen-{}.png'.format(hold_order), dpi=75)
    plt.show()
