import numpy as np

from time import time
from optimize_log import OptimizeLog
from utils import run_optimizer

# https://towardsdatascience.com/unboxing-lasso-regularization-with-proximal-gradient-method-ista-iterative-soft-thresholding-b0797f05f8ea


def prox_l1(w, gamma):
    return np.sign(w) * np.maximum(np.abs(w) - gamma, 0)


def optimize_lasso(oracle, start_point, tol=1e-8, lambda_=1e-2, max_iter=10000, verbose=False):
    log = OptimizeLog(start_time=time())
    
    oracle.reset_oracle_count()
    
    w, L = start_point, 1.0
    for i in range(max_iter):
        value_w, grad_w = oracle.fuse_value_grad(w)
        
        while True:
            w_new = prox_l1(w - (1/L) * grad_w.reshape(-1, 1), (1/L) * lambda_)
            
            w_diff = w_new - w
            w_diff_norm = w_diff.T @ w_diff
            
            if oracle.value(w_new) <= value_w + grad_w @ w_diff + (L / 2) * w_diff_norm:
                break                
            L = 2 * L    
        
        if w_diff_norm * L < tol:
            break
        
        w = w_new
        L = L / 2
        
        if verbose:
            print(f"Iteration {i}: {oracle.value(w_new)}")
        
        log.add_log(time(), value_w, w_diff_norm * L, oracle._call_count)
        
    return w, log


def main():    
    w, log = run_optimizer("data/a1a.txt", optimize_lasso, lambda_=0.00001)
    
    print(log.get_log_last())
    

if __name__ == "__main__":
    main()
    
