import sys
import pickle

from pprint import pprint
import numpy as np

sys.path.append('..')

from collections import defaultdict
from oracle import make_oracle
from optimize_lasso import optimize_lasso
from utils import file_name


def generate_data(data_path, lrs):
    oracle = make_oracle(data_path)
    w_init = np.zeros((oracle.dim, 1))

    res = defaultdict(list)
    res["lr"] = lrs
    res["weights_path"] = np.zeros((oracle.dim, len(lrs)))
     
    best = np.inf
    for i, lr in enumerate(lrs):    
        w, log = optimize_lasso(oracle, w_init, lambda_=lr)
        entropy, n_iter, calls, time, grads = log.get_log_last()
        
        res["weights_path"][:, i] = w.ravel()
        res["time"].append(time)
        res["iters"].append(n_iter)
        res["nonzero"].append(np.sum(w != 0))
        
        if entropy < best:
            best = entropy
            res["best_run_entropy"] = np.array(log.entropy)
            res["best_run_grad"] = np.array(log.grad_info).flatten()
            
    with open(f'{file_name(data_path)}_lr_log.pkl', "bw") as lr_log:
        pickle.dump(res, lr_log)
        
    return res


if __name__ == "__main__":
    generate_data("../data/a1a.txt", np.logspace(-1, -8, 64))
    generate_data("../data/breast-cancer_scale.txt", np.logspace(-1, -8, 64))