import sys
import time
import numpy as np

sys.path.append('..')

import optimize

from oracle import make_oracle
from cross_entropy import sigmoid

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

DELIM = "==" * 35


def _test(data_path=None, optimize_method="optimize", line_search_method="golden", tol=1e-8, verbose=False):
    optimizer = getattr(optimize, optimize_method)
    oracle = make_oracle(data_path=data_path)

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    
    start_time = time.time()
    w_pred = optimizer(oracle, w_init, line_search_method=line_search_method, tol=tol, verbose=verbose)
    end_time = time.time() - start_time
            
    model = LogisticRegression(penalty="none", tol=1e-8, max_iter=10000, n_jobs=-1, fit_intercept=False)
    model.fit(oracle.X, oracle.y.ravel())
    
    print(f"{DELIM}\n"
          f"Test ---- {line_search_method} ---- \n"
          f"data: {data_path}, time: {end_time}\n"
          f"Cross entropy: {oracle.value(w_pred)}\n"
          f"Cross entropy (sklearn): {oracle.value(model.coef_.reshape(-1, 1))}\n"
          f"Entropy difference: {abs(oracle.value(w_pred) - oracle.value(model.coef_.reshape(-1, 1)))}\n"
          f"{DELIM}\n"
        )
    
    
def main():
    # for data in [None, "../data/a1a.txt", "../data/breast-cancer_scale.txt"]:
    #     for method in ["golden", "brent", "armijo", "wolfe", "lipschitz"]:
    #         _test(data_path=data, line_search_method=method) 

    for data in [None, "../data/a1a.txt", "../data/breast-cancer_scale.txt"]:
        for method in ["golden", "brent", "armijo", "wolfe"]:
            _test(data_path=data, optimize_method="optimize_newton", line_search_method=method) 


if __name__ == "__main__":
    main()