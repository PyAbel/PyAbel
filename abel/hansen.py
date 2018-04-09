import numpy as np


def hansen_transform(im, dr=1, direction='inverse', hold_order=1):
    # Hansen IEEE Trans. Acoust. Speech Signal Proc. 33, 666 (1985)
    #  10.1109/TASSP.1985.1164579

    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    # state equation integrals
    def I(n, lam, a):  # integral (epsilon/r)^(lamda+pwr)
        integral = np.empty((n.size, lam.size))
        lama = lam + a

        for k in np.arange(K):
            integral[:, k] = (1 - ratio**lama[k])*(n-1)**a/lama[k]

        return integral

    # special case divide issue for lamda=0, only for inverse transform
    def I0(n, lam, a):
        # Fix me! - uses global variable phi 
        integral = np.empty((n.size, lam.size))

        integral[:, 0] = -np.log(n/(n-1))
        for k in np.arange(1, K):
            integral[:, k] = (1 - phi[:, k])/lam[k]

        return integral

    # first-order hold functions
    def beta0(n, lam, a, intfunc):  # fn   q\epsilon  +  p
        return I(n, lam, a+1) - (n-1)[:, None]*intfunc(n, lam, a)

    def beta1(n, lam, a, intfunc):  # fn-1   p + q\epsilon
        return n[:, None]*intfunc(n, lam, a) - I(n, lam, a+1)

    im = np.atleast_2d(im)

    aim = np.zeros_like(im)  # Abel transform array
    rows, cols = im.shape
    K = h.size

    N = np.arange(cols-1, 1, -1)  # N = cols-1, cols-2, ..., 2
    ratio = N/(N-1)  # cols-1/cols-2, ...,  2/1

    phi = np.empty((N.size, K))
    for k in range(K):
        phi[:, k] = ratio**lam[k]

    if direction == 'forward':
        drive = im.copy()
        h *= -2*dr*np.pi  # include Jacobian with h-array
        a = 1  # integration increases lambda + 1
        intfunc = I
    else:  # inverse Abel transform
        drive = np.gradient(im, dr, axis=-1)
        a = 0  # from 1/piR factor
        intfunc = I0

    x = np.zeros((K, rows))
    if hold_order == 1:  # Hansen first-order hold approximation
        B0 = beta0(N, lam, a, intfunc)*h
        B1 = beta1(N, lam, a, intfunc)*h
        for indx, col in zip(N[::-1]-N[-1], N):
            x = phi[indx][:, None]*x + B0[indx][:, None]*drive[:, col]\
                                     + B1[indx][:, None]*drive[:, col-1]
            aim[:, col-1] = x.sum(axis=0)

    else:  # Hansen (& Law) zero-order hold approximation
        gamma = intfunc(N, lam, a)*h
        for indx, col in zip(N[::-1]-N[-1], N-1):
            x = phi[indx][:, None]*x + gamma[indx][:, None]*drive[:, col]
            aim[:, col] = x.sum(axis=0)

    # missing 1st column
    aim[:, 0] = aim[:, 1]

    if rows == 1:
        aim = aim[0]  # flatter to a vector

    return aim
