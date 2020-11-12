import numpy as np
import step_search

from time import time
from oracle import make_oracle
from optimize_log import OptimizeLog

from optimize_step import descent_step, newton_step, hf_newton_step
    

def _optimize(optimize_step, oracle, w, line_search_method, tol, max_iter, verbose, log_modulo):
    log = OptimizeLog(start_time=time())
    
    line_search = getattr(step_search, f"{line_search_method}_line_search")    
    grad0_norm = np.linalg.norm(oracle.grad(w))**2
    
    for i in range(1, max_iter + 1):
        w, stop_condition, info = optimize_step(
            oracle=oracle, w=w, 
            grad0_norm=grad0_norm, 
            line_search=line_search, 
            tol=tol)
        
        # log: etropy, alpha, grad_info
        if i % log_modulo == 0:            
            entropy, alpha, grad_norm = info
            log.add_log(time(), entropy, alpha, grad_norm)
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}: {entropy}, alpha: {alpha}, grads: {grad_norm}")
        
        if stop_condition:
            break
        
    return w, log


def optimize_gd(oracle, start_point, line_search_method="brent", tol=1e-8, max_iter=10000, verbose=False, log_modulo=100):
    return _optimize(descent_step, oracle, start_point, line_search_method, tol, max_iter, verbose, log_modulo)


def optimize_newton(oracle, start_point, line_search_method="wolfe", tol=1e-8, max_iter=10000, verbose=False, log_modulo=1):
    return _optimize(newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose, log_modulo)


def optimize_hfn(oracle, start_point, line_search_method="armijo", tol=1e-8, max_iter=10000, verbose=False, log_modulo=1):
    return _optimize(hf_newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose, log_modulo)


        
def main():
    oracle = make_oracle("data/a1a.txt")
    # oracle = make_oracle()

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    # w_init = np.random.uniform(size=w_n).reshape(-1, 1)
    # w_init = np.random.normal(size=w_n).reshape(-1, 1)
    # w_init = np.zeros(w_n).reshape(-1, 1)
    # w_init = np.ones(w_n).reshape(-1, 1)

    optimize_hfn(oracle, w_init, "wolfe", tol=1e-16, verbose=True)

    
if __name__ == "__main__":
    main()
    
