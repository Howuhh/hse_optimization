import numpy as np
import step_search

from time import time
from utils import run_optimizer
from optimize_data.optimize_log import OptimizeLog

from optimize_step import descent_step, newton_step, hf_newton_step


def _optimize(optimize_step, oracle, w, line_search_method, tol, max_iter, verbose):
    log = OptimizeLog(start_time=time())
    oracle._call_count = 0  # wellp, thats only for graphs
    
    line_search = getattr(step_search, f"{line_search_method}_line_search")    
    grad0_norm = np.linalg.norm(oracle.grad(w))**2
    
    for i in range(1, max_iter + 1):
        w, stop_condition, info = optimize_step(
            oracle=oracle, w=w, 
            grad0_norm=grad0_norm, 
            line_search=line_search, 
            tol=tol)
        
        # log: etropy, alpha, grad_info
        entropy, alpha, grad_norm = info   
        log.add_log(time(), entropy, alpha, grad_norm, oracle._call_count)
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}: {entropy}, alpha: {alpha}, grads: {grad_norm}")
        
        if stop_condition:
            break
        
    return w, log


def optimize_gd(oracle, start_point, line_search_method="armijo", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(descent_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def optimize_newton(oracle, start_point, line_search_method="wolfe", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def optimize_hfn(oracle, start_point, line_search_method="armijo", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(hf_newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def main():    
    w, log = run_optimizer("data/a1a.txt", optimize_hfn, line_search_method="armijo", verbose=True)
    
    
if __name__ == "__main__":
    main()
    
