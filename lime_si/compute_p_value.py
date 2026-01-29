import numpy as np
from scipy.stats import norm
from . import parametric_lasso
def compute_p_value_naive ( etajTy, tn_mu, tn_sigma):

    cdf_naive = norm.cdf( etajTy ,loc=tn_mu ,scale=tn_sigma)
    val_naive = 2 * min(cdf_naive, 1 - cdf_naive)
    return val_naive

def compute_p_value_para_oc (X_design, y_design, lamda, c_vec, m_vec, etaj, etajTy, num_samples, p, A, bh, cov, tn_miu, tn_sigma):

    theshold = 20*tn_sigma
    list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X_design, y_design, lamda, c_vec, m_vec, etaj, num_samples, p, theshold)

    cdf_val_para, _ = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_miu, tn_sigma, 'A')
    if cdf_val_para is None:
      return None,None
    val_para = 2 * min(cdf_val_para, 1 - cdf_val_para)

    cdf_val_oc, _ = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_miu, tn_sigma, 'As')
    if cdf_val_oc is None:
      return None,None
    val_oc = 2 * min(cdf_val_oc, 1 - cdf_val_oc)

    return val_oc, val_para
