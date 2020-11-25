import numpy as np
import step_search

from time import time
from tabulate import tabulate
from oracle import make_oracle
from optimize_log import OptimizeLog

from optimize_step import descent_step, newton_step, hf_newton_step
    

def _optimize(optimize_step, oracle, w, line_search_method, armijo_init, tol, max_iter, verbose):
    log = OptimizeLog(start_time=time())
    oracle._call_count = 0  # wellp, thats only for graphs
    
    _line_search = getattr(step_search, f"{line_search_method}_line_search")    
    
    # quick fixes for report, not really practical
    if line_search_method == "armijo":
        line_search = lambda oracle, w, direction: _line_search(oracle, w, direction, init_alpha=armijo_init)
    else:
        line_search = _line_search
        
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


def optimize_gd(oracle, start_point, line_search_method="brent", armijo_init="mean", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(descent_step, oracle, start_point, line_search_method, armijo_init, tol, max_iter, verbose)


def optimize_newton(oracle, start_point, line_search_method="wolfe", armijo_init="mean", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(newton_step, oracle, start_point, line_search_method, armijo_init, tol, max_iter, verbose)


def optimize_hfn(oracle, start_point, line_search_method="armijo", armijo_init="mean", tol=1e-8, cg_tol="sqrt", max_iter=10000, verbose=False):
    _hf_newton_step = lambda oracle, w, grad0_norm, line_search, tol: hf_newton_step(oracle, w, grad0_norm, line_search, tol, cg_tol=cg_tol)
    
    return _optimize(_hf_newton_step, oracle, start_point, line_search_method, armijo_init, tol, max_iter, verbose)


def main():    
    oracle = make_oracle("data/a1a.txt")
    # oracle = make_oracle()

    w_n = oracle.X.shape[1]
    # w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    w_init = np.zeros(w_n).reshape(-1, 1)
    # w_init = np.random.uniform(size=w_n).reshape(-1, 1)
    # w_init = np.ones(w_n).reshape(-1, 1)

    w, log = optimize_newton(oracle, w_init, "armijo", tol=1e-8, verbose=True)
    # print(log.get_log()["oracle_calls"][-1])
    
    
if __name__ == "__main__":
    main()
    
