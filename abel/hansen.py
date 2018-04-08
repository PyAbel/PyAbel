# -*- coding: utf-8 -*-
import numpy as np


def hansen_transform(IM, dr=1, direction='inverse', **kwargs):
    def I(n, lam):
        # integral (\epsilon/r)^\lamda d\epsilon
        return 2*(n-1)*(1-(n/(n-1))**lam)/lam

    # corrected beta expressions 10.1109/TASSP.1986.1164860
    def beta0(n, lam):
        # return 2*(n-1)*((n-1) + (l1-n+1)*ratio**l1)/l1/(l1+1)
        return I(n, lam+1) - (n-1)*I(n, lam)

    def beta1(n, lam):
        # return -2*(n-1)*((l1+n) - n*ratio**l1)/l1/(l1+1)
        return n*I(n, lam) - I(n, lam+1)

    # parameters of Abel transform system model, table 1.
    h = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    lam = np.array([0.0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9,
                    -47391.1])

    IM = np.atleast_2d(IM)

    AIM = np.zeros_like(IM)  # forward Abel transform image

    rows, N = IM.shape  # shape of input quadrant (half)
    # enumerate columns n = 0 is Rmax, the right side of image
    n = np.arange(N-1, 0, -1)  # n = N-1, ..., 1
    ratio = n[:-1]/n[1:]  # N-1/(N-2), ..., 2/1

    # phi array Eq (16a), diagonal array, for each pixel
    K = h.size
    phi = np.empty((N-2, K))
    for k in range(K):
        phi[:, k] = ratio**lam[k]

    B0 = np.zeros_like(phi)
    B1 = np.zeros_like(phi)
    for k in range(K):
        B0[:, k] = h[k]*beta0(n[:-1], lam[k])
        B1[:, k] = h[k]*beta1(n[:-1], lam[k])

    # driving function = raw image. Copy so input image not mangled
    if direction=='inverse':
        drive = np.gradient(IM, axis=-1)
    else:
        drive = IM.copy()

    # Hansen and Law Abel transform ---------------
    x = np.zeros((K, rows))
    for indx, col in zip(n[::-1]-1, n[1:]):
        x = phi[indx][:, None]*x + B0[indx][:, None]*drive[:, col+1]\
                                 + B1[indx][:, None]*drive[:, col]
        AIM[:, col] = x.sum(axis=0)

    # missing 1st column
    AIM[:, 0] = AIM[:, 1]

    if AIM.shape[0] == 1:
        AIM = AIM[0]   # flatten to a vector

    print("here!")
    if direction=='inverse':
        return AIM*dr*2
    else:
        return AIM*dr*np.pi
