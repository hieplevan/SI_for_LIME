import numpy as np
from sklearn.linear_model import Lasso

def run_LIME(model, X_test, X_ref, num_samples, p):

    X_test_flat = X_test.flatten()
    Pertub = np.random.randint(0, 2, size=(num_samples, p))

    Xz = Pertub * X_test_flat + (1-Pertub) * X_ref.flatten()
    fz = model.predict(Xz)

    dist = np.linalg.norm(Pertub - np.ones(p), axis=1)
    width = 0.75 * np.sqrt(p)
    weights = np.exp(-(dist**2) / (width**2))
    sqrt_W = np.sqrt(weights)

    Pertub_weighted = Pertub * sqrt_W[:, np.newaxis]
    fz_weighted = fz * sqrt_W

    reg = Lasso(alpha=0.01, fit_intercept=False, tol=1e-10 ,max_iter=10000)
    reg.fit(Pertub_weighted, fz_weighted)

    return reg.coef_, Pertub_weighted, fz_weighted, weights, Pertub