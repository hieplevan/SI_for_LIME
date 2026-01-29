import numpy as np
from lime_si import lime,gen_data,util,compute_p_value

np.random.seed(5)

def run():
    n = 100
    p = 10
    miu = 0
    num_samples = 200
    num_refs = 3
    X_train, y_train  = gen_data.generate_data(n, p)

    X_test= miu + np.random.normal(0, 1, size=p)
    X_refs = miu + np.random.normal(0, 1, size=(num_refs, p))

    X_long = np.concatenate((X_test, X_refs.flatten()))
    X_ref = X_refs.mean(axis=0)

    model, beta_model, bias_model = util.run_model(X_train, y_train)

    bh, X_design, y_design, weights, Pertub = lime.run_LIME(model, X_test, X_ref, num_samples, p)
    cov = np.identity((num_refs+1)*p)

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X_design, bh, p)
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

      s = util.construct_s(bh)
      lamda = 0.01

      tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))
      miuT= np.full((num_refs + 1) * p, miu)
      tn_miu = np.dot(etaj,miuT)
      p_val_oc, p_val_para = compute_p_value.compute_p_value_para_oc(X_design, y_design, lamda, c_vec, m_vec, etaj, etajTy, num_samples, p, A, bh, cov, tn_miu, tn_sigma)
      p_val_naive = compute_p_value.compute_p_value_naive(etajTy,tn_miu, tn_sigma)

      print(f"\n{'='*20} FEATURE SELECTION {j_selected} {'='*20}")
      print(f"   Naive       | P-value: {p_val_naive:.4f}")
      print(f"     OC        | P-value: {p_val_oc:.4f}")
      print(f" Parametric    | P-value: {p_val_para:.4f} ")
if __name__ == "__main__":
  run()


  
