import sys
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import entropy, wasserstein_distance
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
tfd = tfp.distributions

import src.helper as h
from src.model import dirichlet_nll_loss_with_regularization, predictor

exp_seed_str = sys.argv[1]
exp_seed = int(exp_seed_str)
n_classes_str = sys.argv[2]
n_classes = int(n_classes_str)
annotator_num_str = sys.argv[3]
annotator_num = int(annotator_num_str)


#in order not to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
tf.keras.utils.set_random_seed(exp_seed)
tf.config.experimental.enable_op_determinism()

with open(os.path.join("data", "synthetic", "1500_points_"+n_classes_str+"_classes.pkl"), 'rb') as f:
    data = pickle.load(f)



model = predictor(order="first", feature_dim=data["X"].shape[1], n_classes=n_classes)
loss_fn = tf.keras.losses.CategoricalCrossentropy()
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn)
X_train, X_test, lambda_train, lambda_test, lambda_tilde_train, lambda_tilde_test = train_test_split(data["X"], data["W"],  data["W_annotated"][annotator_num], test_size=500, random_state=2024+exp_seed)
X_proper_train, X_calib, lambda_proper_train, lambda_calib, lambda_tilde_proper_train, lambda_tilde_calib = train_test_split(X_train, lambda_train, lambda_tilde_train, test_size=500, random_state=2024+exp_seed)
model.fit(X_proper_train, lambda_tilde_proper_train, epochs=100, batch_size=8)



lambda_hat_calib = model.predict(X_calib)
lambda_hat_test = model.predict(X_test)
distance_functions = ["tv", "kl", "ws", "inner"]
calib_scores = {k: [] for k in distance_functions} #calib scores with lambda_tilde
calib_scores_ideal = {k: [] for k in distance_functions} #calib scores with lambda (just for later comparisons)
for i in range(lambda_hat_calib.shape[0]):
    calib_scores["tv"].append(h.tv(lambda_hat_calib[i], lambda_tilde_calib[i]))
    calib_scores["kl"].append(entropy(lambda_tilde_calib[i], lambda_hat_calib[i], base=2))
    calib_scores["ws"].append(wasserstein_distance(np.arange(n_classes), np.arange(n_classes), lambda_hat_calib[i], lambda_tilde_calib[i]))
    calib_scores["inner"].append(1-np.inner(lambda_hat_calib[i], lambda_tilde_calib[i]))
    calib_scores_ideal["tv"].append(h.tv(lambda_hat_calib[i], lambda_calib[i]))
    calib_scores_ideal["kl"].append(entropy(lambda_calib[i], lambda_hat_calib[i], base=2))
    calib_scores_ideal["ws"].append(wasserstein_distance(np.arange(n_classes), np.arange(n_classes), lambda_hat_calib[i], lambda_calib[i]))
    calib_scores_ideal["inner"].append(1-np.inner(lambda_hat_calib[i], lambda_calib[i]))


q = {}
q_ideal ={}
for d in calib_scores.keys():
    for alpha in [0.05, 0.1, 0.2]:
        q[(d, alpha)] = h.compute_quantile(calib_scores[d], alpha)
        q_ideal[(d, alpha)] = h.compute_quantile(calib_scores_ideal[d], alpha)


cvg = {k: 0 for k in q.keys()}
cvg_ideal = {k: 0 for k in q.keys()}
for i in range(lambda_hat_test.shape[0]):
    for alpha in [0.05, 0.1, 0.2]:
        cvg[("tv", alpha)] += (h.tv(lambda_hat_test[i], lambda_tilde_test[i]) <q[("tv", alpha)])
        cvg[("kl", alpha)] += (entropy(lambda_tilde_test[i], lambda_hat_test[i], base=2) <q[("kl", alpha)])
        cvg[("ws", alpha)] += (wasserstein_distance(np.arange(n_classes), np.arange(n_classes), lambda_hat_test[i], lambda_tilde_test[i]) <q[("ws", alpha)])
        cvg[("inner", alpha)] += ((1-np.inner(lambda_hat_test[i], lambda_tilde_test[i])) <q[("inner", alpha)])

        cvg_ideal[("tv", alpha)] += (h.tv(lambda_hat_test[i], lambda_test[i]) <q[("tv", alpha)])
        cvg_ideal[("kl", alpha)] += (entropy(lambda_test[i], lambda_hat_test[i], base=2) <q[("kl", alpha)])
        cvg_ideal[("ws", alpha)] += (wasserstein_distance(np.arange(n_classes), np.arange(n_classes), lambda_hat_test[i], lambda_test[i]) <q[("ws", alpha)])
        cvg_ideal[("inner", alpha)] += ((1-np.inner(lambda_hat_test[i], lambda_test[i])) <q[("inner", alpha)])

for d in q.keys():
    cvg[d] = cvg[d]/lambda_hat_test.shape[0]
    cvg_ideal[d] = cvg_ideal[d]/lambda_hat_test.shape[0]
    print(d, ": coverage for lambda_tilde is ", cvg[d])
    print(d, ": coverage for lambda is ", cvg_ideal[d])    
data_to_save = {"q":q, "q_ideal":q_ideal, "cvg":cvg, "cvg_ideal":cvg_ideal}





os.makedirs(os.path.join("results", "synthetic", "noise_analysis", "n_classes"+n_classes_str, "annotator_num"+annotator_num_str, exp_seed_str), exist_ok=True)      
with open(os.path.join("results", "synthetic", "noise_analysis", "n_classes"+n_classes_str, "annotator_num"+annotator_num_str, exp_seed_str, "res.pkl"), 'wb') as f:
    pickle.dump(data_to_save, f)


