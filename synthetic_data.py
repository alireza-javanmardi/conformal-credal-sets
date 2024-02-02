import os
import sys
import pickle
import numpy as np

data_num_per_step = sys.argv[1] #number of data points for each train, calibration, test steps
n_classes_str = sys.argv[2]

exp_seed = 200
os.environ['PYTHONHASHSEED']=str(exp_seed)
np.random.seed(exp_seed)

n1 = int(data_num_per_step)
n = 3*n1 
n_classes = int(n_classes_str)

X = np.random.normal(loc=0.0, scale=1.0, size=(n,10)) #random feature vecotrs 
beta = np.random.normal(loc=0.0, scale=1.0, size=(10,n_classes)) #random coefficient vector
Z = np.exp(np.matmul(X,beta)) 
W = Z/Z.sum(axis=1)[:,None] #ground truth distributions associated with Xs

W_annotated = {} #distributions aggregated from annotator_num annotators
for annotator_num in [1, 5, 10, 50, 100, 200, 1000]:

    W_annotated[annotator_num] = np.array([np.random.multinomial(annotator_num, i) for i in W])/annotator_num

data = {
    "X":X,
    "W":W,
    "W_annotated":W_annotated
}
os.makedirs(os.path.join("data", "synthetic"), exist_ok=True)      
with open(os.path.join("data", "synthetic", str(n)+"_points_"+n_classes_str+"_classes.pkl"), 'wb') as f:
    pickle.dump(data, f)