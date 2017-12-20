import numpy as np

#########################################################################
#
# Analytical Abel transform pairs
#  G. C.-Y Chan and G. M. Hieftje Spectrochimica Acta B 61, 31-41 (2006)
#  doi:10.1016/j.sab.2005.11.009
#
# 20-Dec-2017 Stephen Gibson - adapted code for PyAbel
# 20-Nov-2015 Dhrubajyoti Das - python gist 
#             https://github.com/PyAbel/PyAbel/issues/19#issuecomment-158244527
#
#########################################################################


def a(n, x):
    return np.sqrt(n*n - x*x)


def case1(r):
    # Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13) 

    # domain of case functions
    rm = r[r <= 0.25]
    rp = r[r > 0.25]

    em = 3/4 + 12*rm**2 - 32*rm**3
    ep = (16/27)*(1 + 6*rp - 15*rp**2 + 8*rp**3)
    source = np.concatenate((em, ep))

    a4m = a(0.25, rm)
    a1m = a(1, rm)
    rm2 = rm**2
    Im = (128*a1m + a4m)/108 + (283*a4m - 112*a1m)*rm2*2/27 +\
         (4*(1 + rm2)*np.log((1 + a1m)/rm) -\
         (4 + 31*rm2)*np.log((0.25 + a4m)/rm))*rm2*8/9

    a1p = a(1, rp)
    rp2 = rp**2
    Ip = (a1p - 7*a1p*rp2 + 3*rp2*(1 + rp2)*np.log((1 + a1p)/rp))*32/27
    proj = np.concatenate((Im, Ip))

    return source, proj


def case2(r):
    # Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13) 

    source = 1 - 3*r*r + 2*r**3
    a1 = a(1, r)
    proj = a1*(1 - r**2*5/2) + r**4*np.log((1 + a1)/r)*3/2

    return source, proj


def case3(r):
    # Cremers and Birkebak App. Opt. 5, 1057-1064 (1966) Eq(13) 
    # Curve A, Table 2, Fig 3. Hansen&Law JOSA A2 510 (1985)

    # domain regions
    rm = r[r <= 0.5]
    rp = r[r > 0.5]

    em = 1 - 2*rm**2
    ep = 2*(1 - rp)**2
    source = np.concatenate((em, ep))

    a5m = a(0.5, rm)
    a1m = a(1, rm)
    a1p = a(1, rp)

    # power rm**2 typo in Cremers
    Im = (4/3)*a1m*(1 + 2*rm**2) - (2/3)*a5m*(1 + 8*rm**2) -\
          4*rm**2*np.log((1 + a1m)/(0.5 + a5m))

    Ip = (4/3)*a1p*(1 + 2*rp**2) - 4*rp**2*np.log((1 + a1p)/rp)
    proj = np.concatenate((Im, Ip))

    return source, proj


def case4(r):
    #  Alvarez, Rodero, Quintero Spectochim. Acta B 57, 1665-1680 (2002)
    # *invalid*

    rm = r[r <= 0.7]
    rp = r[r > 0.7]

    em = 0.1 + 5.51*rm**2 - 5.25*rm**3
    ep = -40.74 + 155.6*rp - 188.89*rp**2 + 74.07*rp**3
    source = np.concatenate((em, ep))

    a7m = a(0.7, rm)
    a1m = a(1, rm)
    a7p = a(0.7, rp)
    a1p = a(1, rp)

    Im = 22.68862*a7m - 14.811667*a1m + (217.557*a7m - 193.30083*a1m)*rm**2 +\
         155.56*rm**2*np.log((1 + a1m)/(0.7 + a7m)) +\
         rm**4*(55.5525*np.log((1 + a1m)/rm) - 59.49*np.log((0.7 + a7m)/rm))

    Ip = -14.811667*a1p - 193.0083*a1p*rp**2 +\
         rp**2*(155.56 + 55.5525*rp**2)*np.log((1 + a1p)/rp)
    proj = np.concatenate((Im, Ip))

    return source, proj


def case5(r):
    # Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996)

    source = np.ones_like(r)
    proj = 2*a(1, r)

    return source, proj


def case6(r):
    #  Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996)

    source = np.exp(1.1**2*(1 - 1/(1 - r**2)))/np.sqrt(1 - r**2)**3
    proj = np.exp(1.1**2*(1 - 1/(1 - r**2)))*np.sqrt(np.pi)/1.1/a(1, r)

    return source, proj


def case7(r):
    #  Buie et al. J. Quant. Spectrosc. Radiat. Transfer 55, 231-243 (1996

    source = (1 + 10*r**2 - 23*r**4 + 12*r**6)/2
    proj = a(1, r)*(19 + 34*r**2 - 125*r**4 + 72*r**6)*8/105

    return source, proj
