import scipy
import numpy as np
import step_search

from oracle import make_oracle
from utils import shift_positive_definite, solve_cholesky


def _optimize(optimize_step, oracle, start_point, line_search_method, tol, max_iter, verbose):
    line_search = getattr(step_search, f"{line_search_method}_line_search")
    
    grad0_norm = np.linalg.norm(oracle.grad(start_point))**2
    
    w = start_point
    for i in range(1, max_iter):
        w, stop_condition = optimize_step(
            oracle, w, 
            grad0_norm=grad0_norm, 
            iter_num=i, 
            line_search=line_search, 
            tol=tol, 
            verbose=verbose
        )
        
        if stop_condition:
            break
        
    return w
    

def descent_step(oracle, w, grad0_norm, iter_num, line_search, tol, verbose):
    fi, grad = oracle.fuse_value_grad(w)
    direction = grad.reshape(-1, 1)

    assert(w.shape == direction.shape), "diff shape for grad and w"
    
    alpha = line_search(oracle, w, direction)
    w = w - alpha * direction
    
    grad_ratio = np.linalg.norm(grad)**2 / grad0_norm
    
    if verbose and iter_num % 100 == 0:
        print(f"Iteration {iter_num}: {fi}, alpha: {alpha}, grads: {grad_ratio}")
    
    return w, (grad_ratio <= tol)


def newton_step(oracle, w, grad0_norm, iter_num, line_search, tol, verbose):
    fi, grad, hessian = oracle.fuse_value_grad_hessian(w)
    
    pos_hessian = shift_positive_definite(hessian)
    direction = solve_cholesky(pos_hessian, grad.reshape(-1, 1))
    # direction = (grad @ np.linalg.inv(pos_hessian)).reshape(-1, 1)

    assert(w.shape == direction.shape), "diff shape for grad and w"
    
    alpha = line_search(oracle, w, direction)
    w = w - alpha * direction
    
    grad_ratio = np.linalg.norm(grad)**2 / grad0_norm
    
    if verbose and iter_num % 1 == 0:
        print(f"Iteration {iter_num}: {fi}, alpha: {alpha}, grads: {grad_ratio}")
    
    return w, (grad_ratio <= tol)


def optimize(oracle, start_point, line_search_method="brent", tol=1e-8, max_iter=10000, verbose=True):
    return _optimize(descent_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def optimize_newton(oracle, start_point, line_search_method="brent", tol=1e-8, max_iter=10000, verbose=True):
    return _optimize(newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose)

        
def main():
    oracle = make_oracle("data/a1a.txt")

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    # w_init = np.zeros(w_n).reshape(-1, 1)
    
    optimize_newton(oracle, w_init, line_search_method="wolfe")

    
if __name__ == "__main__":
    main()