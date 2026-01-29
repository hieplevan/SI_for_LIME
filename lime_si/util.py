import numpy as np
from sklearn.linear_model import LinearRegression

def compute_quotient(numerator, denominator):
    if denominator == 0:
        return np.inf

    quotient = numerator / denominator

    if quotient <= 0:
        return np.inf

    return quotient

def construct_test_statistic(X_long, j, p, num_refs):
    etaj = np.zeros((num_refs + 1) * p)
    etaj[j] = 1.0      
    for i in range(num_refs):
        etaj[p + i * p + j] = -1.0 / num_refs
    T = np.dot(etaj, X_long)
    return etaj, T

def construct_m_c_from_model(X_long, etaj, beta_model, bias_model, Pertub, p, num_refs):

    Cov = np.identity((num_refs + 1) * p)

    denom = np.dot(etaj, np.dot(Cov, etaj))
    if denom == 0 :
      return None,None
    b = np.dot(Cov, etaj) / denom
    z = np.dot(etaj, X_long)
    a = X_long - b * z
    a_test = a[:p]
    b_test = b[:p]

    a_refs = a[p:].reshape(num_refs, p)
    b_refs = b[p:].reshape(num_refs, p)

    a_ref_mean = a_refs.mean(axis=0)
    b_ref_mean = b_refs.mean(axis=0)

    X_comb_a = Pertub * a_test + (1 - Pertub) * a_ref_mean
    X_comb_b = Pertub * b_test + (1 - Pertub) * b_ref_mean

    c = np.dot(X_comb_a, beta_model) + bias_model
    m = np.dot(X_comb_b, beta_model)
    return c, m

def run_model(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    beta_model = model.coef_
    bias_model = model.intercept_

    return model, beta_model, bias_model

def construct_A_XA_Ac_XAc_bhA(X, bh, p):
    A = []
    Ac = []
    bhA = []
    for j in range(p):
        bhj = bh[j]
        if bhj != 0:
            A.append(j)
            bhA.append(bhj)
        else:
            Ac.append(j)
    XA = X[:, A]
    XAc = X[:, Ac]
    bhA = np.array(bhA).reshape((len(A), 1))
    return A, XA, Ac, XAc, bhA

def construct_s(bh):
    s = []
    for bhj in bh:
        if bhj != 0:
            s.append(np.sign(bhj))
    s = np.array(s).reshape((len(s), 1))
    return s
