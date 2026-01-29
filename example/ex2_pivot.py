import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import kstest
import statsmodels.api as sm
from lime_si import lime,gen_data,util,compute_p_value

np.random.seed(5)

def run(X_train, y_train, X_ref, X_test, X_long, miu , num_samples, num_refs, p):
    model, beta_model, bias_model = util.run_model(X_train, y_train)

    bh, X_design, y_design, weights, Pertub = lime.run_LIME(model, X_test, X_ref, num_samples, p)
    cov = np.identity((num_refs+1)*p)

    A, XA, Ac, XAc, bhA = util.construct_A_XA_Ac_XAc_bhA(X_design, bh, p)
    if len(A) == 0 :
      return None
    null_features = [j for j in A ]
    if len(null_features) == 0:
      return None

    rand_index = np.random.randint(len(null_features))
    j_selected = null_features[rand_index]
    etaj, etajTy = util.construct_test_statistic(X_long, j_selected , p, num_refs)


    c_raw, m_raw = util.construct_m_c_from_model(X_long, etaj, beta_model, bias_model, Pertub, p, num_refs)
    if c_raw is None or m_raw is None:
      return None

    sqrt_W = np.sqrt(weights)
    m_vec = (m_raw * sqrt_W).reshape(-1, 1)
    c_vec = (c_raw * sqrt_W).reshape(-1, 1)

    s = util.construct_s(bh)
    lamda = 0.01

    tn_sigma = np.sqrt(np.dot(np.dot(etaj.T, cov), etaj))
    miuT= np.full((num_refs + 1) * p, miu)
    tn_miu = np.dot(etaj,miuT)
    val_SI, val_para = compute_p_value.compute_p_value_para_oc(X_design, y_design, lamda, c_vec, m_vec, etaj, etajTy, num_samples, p, A, bh, cov, tn_miu, tn_sigma)
    val_naive = compute_p_value.compute_p_value_naive(etajTy,tn_miu, tn_sigma)

    return val_SI, val_naive, val_para

if __name__ == "__main__":

    n = 100
    p = 10
    miu = 0
    num_samples = 200
    alpha_level= 0.05
    num_refs = 3
    X_train, y_train  = gen_data.generate_data(n, p)

    list_p_value_oc = []
    list_p_value_naive = []
    list_p_value_para = []

    for i in tqdm(range(1000)):
        X_test = miu + np.random.normal(0, 1, size=p)
        X_refs = miu + np.random.normal(0, 1, size=(num_refs, p))

        X_long = np.concatenate((X_test, X_refs.flatten()))
        X_ref = X_refs.mean(axis=0)


        result = run (X_train, y_train, X_ref, X_test, X_long, miu, num_samples, num_refs, p)

        if result is not None:
            list_p_value_oc.append(result[0])
            list_p_value_naive.append(result[1])
            list_p_value_para.append(result[2])

    stat, p_ks = kstest(list_p_value_para, 'uniform')
    print(f"KS Test P-value: {p_ks:.4f}")
    if p_ks > 0.05:
        print(" P-value phân phối đều.")
    else:
        print(" Not Uniform.")

    plt.rcParams.update({'font.size': 18})
    grid = np.linspace(0, 1, 101)
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value_para))(grid), 'r-', linewidth=6, label='p-value_SI')
    plt.plot(grid, sm.distributions.ECDF(np.array(list_p_value_naive))(grid), 'b-', linewidth=6, label='p-value_Naive')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.hist(list_p_value_para)
    plt.show()
    plt.figure(figsize=(8, 6))

