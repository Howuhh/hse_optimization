import sys
import pickle
import numpy as np

sys.path.append('..')
np.random.seed(42)

from tqdm import tqdm
from collections import defaultdict
from oracle import make_oracle

from optimize import optimize_gd, optimize_hfn, optimize_newton
from optimize_bfgs import optimize_bfgs, optimize_lbfgs
from sklearn.linear_model import LogisticRegression


def file_name(path):
    if path is None:
        return "synthetic"
    return path.split("/")[-1].split(".")[0] 


def optimize_sklearn(oracle):
    model = LogisticRegression(penalty="none", tol=1e-8, max_iter=10000, n_jobs=-1, fit_intercept=False)
    model.fit(oracle.X, oracle.y.ravel())
    
    entropy_true = oracle.value(model.coef_.reshape(-1, 1))
    return entropy_true


def run_model(config, data_paths, log_path="."):
    log_data = defaultdict(dict)
    
    for data_path in tqdm(data_paths):
        oracle = make_oracle(data_path=data_path)
        entropy_true = optimize_sklearn(oracle)
        w_init = np.zeros((oracle.X.shape[1], 1))
                        
        data_name = file_name(data_path)

        for method in tqdm(config, desc=f"Data -- {data_path}"):
            optimizer, params = config[method]["optimizer"], config[method]["params"]            
            w, log = optimizer(oracle, w_init, **params)
            
            log.error = np.abs(log.get_log()["entropy"] - entropy_true)
            
            log_data[data_name][method] = log

    with open(log_path, "wb") as log_file:
        pickle.dump(log_data, log_file)          
      
    return log_data
    
    
def all_methods():
    data_paths = [None, "../data/a1a.txt", "../data/breast-cancer_scale.txt"]
    config = {
        "Gradient Descent": {
            "optimizer": optimize_gd,
            "params": {
                "line_search_method": "armijo"
            }
        },
        "Newton": {
            "optimizer": optimize_newton,
            "params": {
                "line_search_method": "wolfe"
            }
        },
        "Hessian-Free Newton": {
            "optimizer": optimize_hfn,
            "params": {
                "line_search_method": "armijo"
            }
        },
        "BFGS": {
            "optimizer": optimize_bfgs,
            "params": {
                "gamma": 20.0
            }
        },
        "L-BFGS": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 100,
                "gamma": 1.0
            }
        }
    }
    
    log_data = run_model(config, data_paths, "all_methods.pkl")
    
    
def buffer_size():
    data_paths = [None, "../data/a1a.txt", "../data/breast-cancer_scale.txt"]
    config = {
        "L-BFGS (buffer=5)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 5,
                "gamma": 1.0
            }
        },
        "L-BFGS (bufffer=10)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 10,
                "gamma": 1.0
            }
        },
        "L-BFGS (buffer=20)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 20,
                "gamma": 1.0
            }
        },
        "L-BFGS (buffer=50)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 50,
                "gamma": 1.0
            }
        },
        "L-BFGS (buffer=100)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 100,
                "gamma": 1.0
            }
        }
    }
    
    log_data = run_model(config, data_paths, "buffer_size.pkl")


if __name__ == "__main__":
    all_methods()
    buffer_size()