import numpy as np
from sklearn import linear_model
from . import util
from mpmath import mp



def parametric_lasso(X, yz, lamda, b, n, p):
    yz_flatten = yz.flatten()

    clf = linear_model.Lasso(alpha=lamda, fit_intercept=False, tol=1e-10)
    clf.fit(X, yz_flatten)
    bhz = clf.coef_

    Az, XAz, Acz, XAcz, bhAz = util.construct_A_XA_Ac_XAc_bhA(X, bhz, p)

    etaAz = np.array([])

    if XAz is not None:
        inv = np.linalg.pinv(np.dot(XAz.T, XAz))
        invXAzT = np.dot(inv, XAz.T)
        etaAz = np.dot(invXAzT, b)

    shAz = np.array([])
    gammaAz = np.array([])

    if XAcz is not None:
        if XAz is None:
            e1 = yz
        else:
            e1 = yz - np.dot(XAz, bhAz)

        e2 = np.dot(XAcz.T, e1)
        shAz = e2/(lamda * n)

        if XAz is None:
            gammaAz = (np.dot(XAcz.T, b)) / n
        else:
            gammaAz = (np.dot(XAcz.T, b) - np.dot(np.dot(XAcz.T, XAz), etaAz)) / n

    bhAz = bhAz.flatten()
    etaAz = etaAz.flatten()
    shAz = shAz.flatten()
    gammaAz = gammaAz.flatten()

    min1 = np.inf
    min2 = np.inf

    for j in range(len(etaAz)):
        numerator = - bhAz[j]
        denominator = etaAz[j]

        quotient = util.compute_quotient(numerator, denominator)

        if quotient < min1:
            min1 = quotient

    for j in range(len(gammaAz)):
        numerator = (np.sign(gammaAz[j]) - shAz[j])*lamda
        denominator = gammaAz[j]

        quotient = util.compute_quotient(numerator, denominator)
        if quotient < min2:
            min2 = quotient

    return min(min1, min2), Az, bhz

def run_parametric_lasso(X, y, lamda, c , m , etaj, n, p, threshold):

    zk = -threshold
    list_zk = [zk]
    list_active_set = []
    list_bhz = []

    while zk < threshold:
        yz = np.dot(m,zk) + c

        skz, Akz, bhkz = parametric_lasso(X, yz, lamda, m, n, p)

        zk = zk + skz + 0.0001

        if zk < threshold:
            list_zk.append(zk)
        else:
            list_zk.append(threshold)

        list_active_set.append(Akz)
        list_bhz.append(bhkz)
    return list_zk, list_bhz, list_active_set


def pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_mu, tn_sigma, type):

    z_interval = []

    for i in range(len(list_active_set)):
        if type == 'As':
            if np.array_equal(np.sign(bh), np.sign(list_bhz[i])):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

        if type == 'A':
            if np.array_equal(A, list_active_set[i]):
                z_interval.append([list_zk[i], list_zk[i + 1] - 1e-10])

    new_z_interval = []

    for each_interval in z_interval:
        if len(new_z_interval) == 0:
            new_z_interval.append(each_interval)
        else:
            sub = each_interval[0] - new_z_interval[-1][1]
            if abs(sub) < 0.01:
                new_z_interval[-1][1] = each_interval[1]
            else:
                new_z_interval.append(each_interval)

    z_interval = new_z_interval
    numerator = 0
    denominator = 0
    for each_interval in z_interval:
        al = each_interval[0]
        ar = each_interval[1]

        denominator = denominator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

        if etajTy >= ar:
            numerator = numerator + mp.ncdf((ar - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)
        elif (etajTy >= al) and (etajTy < ar):
            numerator = numerator + mp.ncdf((etajTy - tn_mu)/tn_sigma) - mp.ncdf((al - tn_mu)/tn_sigma)

    if denominator != 0:
        return float(numerator/denominator), z_interval
    else:
        return None
