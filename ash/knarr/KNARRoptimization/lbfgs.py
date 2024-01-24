import numpy as np
import KNARRsettings

# Author: Vilhjalmur Asgeirsson, 2019

def LBFGSUpdate(R, R0, F, F0, sk, yk, rhok, memory):
    dr = R - R0
    df = -(F - F0)
    if abs(np.dot(dr.T, df)) < 1e-30:
        raise ZeroDivisionError()
    else:
        sk.append(dr)
        yk.append(df)
        rhok.append(1.0 / np.dot(dr.T, df))

    if len(sk) > memory:
        sk.pop(0)
        yk.pop(0)
        rhok.pop(0)

    return sk, yk, rhok


def LBFGSStep(F, sk, yk, rhok):
    neg_curv = False
    C = rhok[-1] * np.dot(yk[-1].T, yk[-1])
    H0 = 1.0 / C
    if H0 < 0.0:
        if KNARRsettings.printlevel > 0:
            print('**Warning: Negative curvature. Restarting optimizer.')
        neg_curv = True

    lengd = len(sk)
    q = -F.copy()
    alpha = np.zeros(shape=(lengd, 1))

    for i in range(lengd - 1, -1, -1):
        alpha[i] = rhok[i] * np.dot(sk[i].T, q)
        q = q - (alpha[i] * yk[i])

    r = H0 * q

    for i in range(0, lengd):
        beta = rhok[i] * np.dot(yk[i].T, r)
        r = r + sk[i] * (alpha[i] - beta)

    step = -r
    return step, neg_curv
