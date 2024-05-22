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
predictor_order = sys.argv[2]

#in order not to use GPU
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
tf.keras.utils.set_random_seed(exp_seed)
tf.config.experimental.enable_op_determinism()

with open(os.path.join("data", "chaosNLI", "embeddings", "snli.pkl"), 'rb') as f:
    snli = pickle.load(f)

with open(os.path.join("data", "chaosNLI", "embeddings", "mnli_m.pkl"), 'rb') as f:
    mnli = pickle.load(f)

embedding = np.concatenate((snli["embedding"], mnli["embedding"]), axis=0)
premise = np.concatenate((snli["premise"], mnli["premise"]), axis=0)
hypothesis = np.concatenate((snli["hypothesis"], mnli["hypothesis"]), axis=0)
label_dist = np.concatenate((snli["label_dist"], mnli["label_dist"]), axis=0)   
# label_dist_rounded = h.prob_rounder(label_dist, 3)

simplex = np.load(os.path.join("data", "simplex", "0.001.npy"))

model = predictor(order=predictor_order)
if predictor_order == "first": 
    loss_fn = tf.keras.losses.CategoricalCrossentropy()
else: 
    loss_fn = dirichlet_nll_loss_with_regularization
model.compile(optimizer=Adam(learning_rate=1e-4), loss=loss_fn)
X_train, X_test, lambda_train, lambda_test, premise_train, premise_test, hypothesis_train, hypothesis_test = train_test_split(embedding, label_dist,  premise, hypothesis, test_size=500, random_state=2024+exp_seed)
X_proper_train, X_calib, lambda_proper_train, lambda_calib, premise_proper_train, premise_calib, hypothesis_proper_train, hypothesis_calib = train_test_split(X_train, lambda_train, premise_train, hypothesis_train, test_size=500, random_state=2024+exp_seed)
model.fit(X_proper_train, lambda_proper_train, epochs=100, batch_size=8)


if predictor_order == "first": 
    lambda_hat_calib = model.predict(X_calib)
    lambda_hat_test = model.predict(X_test)
    distance_functions = ["tv", "kl", "ws", "inner"]
    calib_scores = {k: [] for k in distance_functions}
    for i in range(lambda_hat_calib.shape[0]):
        calib_scores["tv"].append(h.tv(lambda_hat_calib[i], lambda_calib[i]))
        calib_scores["kl"].append(entropy(lambda_calib[i], lambda_hat_calib[i], base=2))
        calib_scores["ws"].append(wasserstein_distance([0,1,2], [0,1,2], lambda_hat_calib[i], lambda_calib[i]))
        calib_scores["inner"].append(1-np.inner(lambda_hat_calib[i], lambda_calib[i]))
    data_to_save = {"calib_score":calib_scores, "lambda_hat_calib":lambda_hat_calib, "lambda_hat_test":lambda_hat_test, "lambda_test":lambda_test, "premise_test": premise_test, "hypothesis_test": hypothesis_test}


else:
    alpha_calib = model.predict(X_calib) + 1 
    alpha_test = model.predict(X_test) + 1
    mode_calib = (alpha_calib-1)/np.sum(alpha_calib-1, axis=1)[:,None]
    mode_test = (alpha_test-1)/np.sum(alpha_test-1, axis=1)[:,None]
    dist_lambda_hat_calib = tfd.Dirichlet(alpha_calib)
    calib_scores = []
    for k in range(X_calib.shape[0]):
        # probs = (dist_lambda_hat_calib[k].prob(simplex)).numpy()
        # p = dist_lambda_hat_calib[k].prob(lambda_calib[k]).numpy()
        # calib_scores.append(1-p/(np.max(probs)))
        mode_p = dist_lambda_hat_calib[k].prob(mode_calib[k]).numpy()
        p = dist_lambda_hat_calib[k].prob(lambda_calib[k]).numpy()
        calib_scores.append(1-p/mode_p)
    data_to_save = {"calib_score":calib_scores, "alpha_test":alpha_test, "lambda_test":lambda_test, "mode_test": mode_test, "premise_test": premise_test, "hypothesis_test": hypothesis_test}


os.makedirs(os.path.join("results", "chaosNLI", exp_seed_str), exist_ok=True)      
with open(os.path.join("results", "chaosNLI", exp_seed_str, predictor_order+"_order_data.pkl"), 'wb') as f:
    pickle.dump(data_to_save, f)


