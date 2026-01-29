import numpy as np
from lime_si import lime,gen_data,util
from lime_si import parametric_lasso
np.random.seed(5)

def run():
    n = 100
    p = 10
    miu = 0
    num_samples = 200
    num_refs = 3
    X_train, y_train  = gen_data.generate_data(n, p)

    X_test = miu + np.random.normal(0, 1, size=p)
    X_refs = miu + np.random.normal(0, 1, size=(num_refs, p))

    X_long = np.concatenate((X_test, X_refs.flatten()))
    X_ref = X_refs.mean(axis=0)
    
    model, beta_model, bias_model = util.run_model(X_train, y_train)

    bh, X_design, y_design, weights, Pertub = lime.run_LIME(model, X_test, X_ref, num_samples, p)
    cov = np.identity((num_refs+1)*p)

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X_design, bh, p)
    s = util.construct_s(bh)
    lamda = 0.01
    if len(A) == 0 :
      return None
    null_features = [j for j in A ]
    if len(null_features) == 0:
      return None
    for j_selected in null_features:
        etaj, etajTy = util.construct_test_statistic(X_long, j_selected , p, num_refs)
        c_raw, m_raw = util.construct_m_c_from_model(X_long, etaj, beta_model, bias_model, Pertub, p, num_refs)
        if c_raw is None or m_raw is None:
          continue

        sqrt_W = np.sqrt(weights)
        m_vec = (m_raw * sqrt_W).reshape(-1, 1)
        c_vec = (c_raw * sqrt_W).reshape(-1, 1)


        tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))
        miuT= np.full((num_refs + 1) * p, miu)
        tn_miu = np.dot(etaj,miuT)

        theshold = 20*tn_sigma
        list_zk, list_bhz, list_active_set = parametric_lasso.run_parametric_lasso(X_design, y_design, lamda, c_vec, m_vec, etaj, num_samples, p, theshold)

        cdf_val_para, z_para_interval = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_miu, tn_sigma, 'A')
        if cdf_val_para is None:
          continue
        val_para = 2 * min(cdf_val_para, 1 - cdf_val_para)

        cdf_val_oc, z_oc_interval = parametric_lasso.pivot(A, bh, list_active_set, list_zk, list_bhz, etaj, etajTy, cov, tn_miu, tn_sigma, 'As')
        if cdf_val_oc is None:
          continue
        val_oc = 2 * min(cdf_val_oc, 1 - cdf_val_oc)
        print(f"\n{'='*20} FEATURE SELECTION {j_selected} {'='*20}")
        print(f"     OC        | P-value: {val_oc:.4f} | Interval: {z_oc_interval}")
        print(f" Parametric    | P-value: {val_para:.4f} | Interval: {z_para_interval}")



if __name__ == "__main__":
    run()

 