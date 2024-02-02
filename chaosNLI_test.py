import sys
import os
import pickle
import numpy as np
from scipy.stats import entropy, wasserstein_distance
import src.helper as h
import tensorflow_probability as tfp
tfd = tfp.distributions

exp_seed_str = sys.argv[1]
exp_seed = int(exp_seed_str)
alpha_str = sys.argv[2]
alpha = float(alpha_str)



simplex_res = 0.005
simplex = np.load(os.path.join("data", "simplex", str(simplex_res)+".npy"))

with open(os.path.join("results", "chaosNLI", exp_seed_str, "first_order_data.pkl"), 'rb') as f:
        fo_data = pickle.load(f)
with open(os.path.join("results", "chaosNLI", exp_seed_str, "second_order_data.pkl"), 'rb') as f:
        so_data = pickle.load(f)

calib_scores = fo_data["calib_score"]
calib_scores["so"] = so_data["calib_score"]

lambda_hat_test, lambda_test, alpha_test= fo_data["lambda_hat_test"] , fo_data["lambda_test"], so_data["alpha_test"]
dist_pred_test = tfd.Dirichlet(alpha_test)

q = {}
for d in calib_scores.keys():
    q[d] = h.compute_quantile(calib_scores[d], alpha)
    # print(d, q[d])


cvg = {k: 0 for k in calib_scores.keys()}
for i in range(lambda_hat_test.shape[0]):
    cvg["tv"] += (h.tv(lambda_hat_test[i], lambda_test[i]) <q["tv"])
    cvg["kl"] += (entropy(lambda_test[i], lambda_hat_test[i], base=2) <q["kl"])
    cvg["ws"] += (wasserstein_distance([0,1,2], [0,1,2], lambda_hat_test[i], lambda_test[i]) <q["ws"])
    cvg["inner"] += ((1-np.inner(lambda_hat_test[i], lambda_test[i])) <q["inner"])

    probs = (dist_pred_test[i].prob(simplex)).numpy()
    p = dist_pred_test[i].prob(lambda_test[i]).numpy()
    cvg["so"] += p/(np.max(probs)) >= (1-q["so"])

for d in calib_scores.keys():
    cvg[d] = cvg[d]/lambda_hat_test.shape[0]
    print(d, ": coverage is ", cvg[d])

os.makedirs(os.path.join("results", "chaosNLI",exp_seed_str, "simplex_res_"+str(simplex_res), alpha_str), exist_ok=True)      
with open(os.path.join("results", "chaosNLI", exp_seed_str, "simplex_res_"+str(simplex_res), alpha_str, "cvg.pkl"), 'wb') as f:
    pickle.dump(cvg, f)

idx_set_test = {k: [] for k in calib_scores.keys()}
set_size_test = {k: [] for k in calib_scores.keys()}
for k in range(lambda_hat_test.shape[0]):
    ph = lambda_hat_test[k]
    idx_tv = np.where((0.5*np.sum(np.abs(ph-simplex), axis=1))< q["tv"])[0]
    idx_set_test["tv"].append(idx_tv)
    set_size_test["tv"].append(len(idx_tv))
    idx_kl = np.where(entropy(simplex, ph, base=2, axis=1)< q["kl"])[0]
    idx_set_test["kl"].append(idx_kl)
    set_size_test["kl"].append(len(idx_kl))
    idx_inner = np.where((1-np.inner(ph, simplex))< q["inner"])[0]
    idx_set_test["inner"].append(idx_inner)
    set_size_test["inner"].append(len(idx_inner))

    probs = (dist_pred_test[k].prob(simplex)).numpy()
    ix_so = np.where(probs/np.max(probs) >= (1-q["so"]))[0]
    idx_set_test["so"].append(ix_so)
    set_size_test["so"].append(len(ix_so))

    idx_ws = []
    ws_dis = []
    for i in range(simplex.shape[0]):
        if wasserstein_distance([0,1,2], [0,1,2], ph, simplex[i]) < q["ws"]:
            idx_ws.append(i)
    idx_set_test["ws"].append(idx_ws)
    set_size_test["ws"].append(len(idx_ws))
    # print(100*k/lambda_hat_test.shape[0])
     
with open(os.path.join("results", "chaosNLI", exp_seed_str, "simplex_res_"+str(simplex_res), alpha_str, "idx_set_test.pkl"), 'wb') as f:
    pickle.dump(idx_set_test, f)
with open(os.path.join("results", "chaosNLI", exp_seed_str, "simplex_res_"+str(simplex_res), alpha_str, "set_size_test.pkl"), 'wb') as f:
    pickle.dump(set_size_test, f)