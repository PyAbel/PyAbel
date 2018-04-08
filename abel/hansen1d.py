import numpy as np

def hansen_transform(IM, dr=1, direction='inverse', hold_order=1):
    # Hansen IEEE Trans. Acoust. Speech Signal Proc. 33, 666 (1985)
    #  10.1109/TASSP.1985.1164579

    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    def phi(n, lam):
        return (n/(n-1))**lam

    def I(n, lam, pwr):  # integral (epsilon/r)^{lamda+pwr} 
        if pwr != -1:
            pwr1 = pwr + 1  # integration +1 to power of lambda
            lp = lam + pwr1  
            integral = 2*(n-1)**pwr1*(1 - phi(n, lp))/lp

        else:  # special case divide issue for lamda=0
            integral = np.empty_like(lam)

            integral[0] = -np.log(n/(n-1))
            integral[1:] = (1 - phi(n, lam[1:]))/lam[1:]

        return integral

    # beta0, beta1, first-order hold functions
    def beta0(n, lam, pwr):  # fn   q\epsilon  +  p
        return I(n, lam, 1+pwr) - (n-1)*I(n, lam, pwr)

    def beta1(n, lam, pwr):  # fn-1   p + q\epsilon
        return n*I(n, lam, pwr) - I(n, lam, 1+pwr)

    if direction == 'forward':
        drive = IM.copy()
        h *= -dr*np.pi  # include Jacobian with h-array
        pwr = 0

    else:  # inverse Abel transform
        drive = np.gradient(IM, dr)
        pwr = -1  # due to 1/piR factor

    AIM = np.zeros_like(IM)  # forward Abel transform image
    cols = IM.shape[-1]
    N = np.arange(cols-1, 1, -1)

    x = np.zeros((h.size, 1))

    if hold_order==0:  # Hansen & Law zero-order hold approximation
        for n in N:
            x  = phi(n, lam)[:, None]*x\
                 + (I(n, lam, pwr)*h)[:, None]*drive[n-1]

            AIM[n-1] = x.sum()

    else:  # Hansen first-order hold approximation
        for n in N:
            x  = phi(n, lam)[:, None]*x\
                 + (beta0(n, lam, pwr)*h)[:, None]*drive[n]\
                 + (beta1(n, lam, pwr)*h)[:, None]*drive[n-1]

            AIM[n-1] = x.sum()

    # missing 1st column
    AIM[0] = AIM[1]

    return AIM

if __name__ == '__main__':

    import abel
    import matplotlib.pyplot as plt

    n = 101
    hold_order = 1

    f = abel.tools.analytical.TransformPair(n, 3)

    hf = hansen_transform(f.func, dr=f.dr, direction='forward',
                          hold_order=hold_order)
    msef = np.square(hf-f.abel).mean()

    hi = hansen_transform(f.func, dr=f.dr, direction='inverse',
                          hold_order=hold_order)
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
    ax1.annotate('mse={:4.2e}'.format(msei), (0.6, 0.5))
    ax1.set_xlabel(r'radius')
    ax1.set_ylabel(r'source')
    ax1.set_title('inverse')

    plt.suptitle(r'Hansen {:s}-order hold, curve A ({:s}), N={:d}'.\
                 format('first' if hold_order else 'zero', f.label, n))
    plt.subplots_adjust(wspace=0.4)
    plt.savefig('hansen-{}.png'.format(hold_order), dpi=75)
    plt.show() 
