import sys
import pickle
import numpy as np

sys.path.append('..')
np.random.seed(42)

from tqdm import tqdm
from collections import defaultdict
from oracle import make_oracle

from optimize import optimize_gd, optimize_hfn, optimize_newton
from sklearn.linear_model import LogisticRegression


def file_name(path):
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
                        
        if data_path is None:
            data_name = "generated"
        else:
            data_name = file_name(data_path)

        for method in tqdm(config, desc=f"Data -- {data_path}"):
            optimizer = config[method]["optimizer"]
            line_search_methods = config[method]["line_search_methods"]
            
            log_data[data_name][method] = {}
        
            for line_search in tqdm(line_search_methods, desc=f"Method -- {method}"):
                w, log = optimizer(oracle, w_init, line_search, tol=1e-8, max_iter=10000, verbose=False)
                
                log.error = np.abs(log.get_log()["entropy"] - entropy_true)
                log_data[data_name][method][line_search] = log

    with open(log_path, "wb") as log_file:
        pickle.dump(log_data, log_file)          
      
    return log_data
    
    
def main():
    data_paths = [None, "../data/a1a.txt", "../data/breast-cancer_scale.txt"]
    config = {
        "gradient descent": {
            "optimizer": optimize_gd,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe", "lipschitz"]
        },
        "newton": {
            "optimizer": optimize_newton,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe"]
        },
        "hf_newton": {
            "optimizer": optimize_hfn,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe"]
        }
    }
    
    log_data = run_model(config, data_paths, "line_search_data.pkl")
    
    
if __name__ == "__main__":
    main()
    
    