import scipy
import numpy as np
import step_search

from oracle import make_oracle
from utils import shift_positive_definite, solve_cholesky, conjugate_grad


def optimize(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000, verbose=True):
    line_search = getattr(step_search, f"{line_search_method}_line_search")
    
    grad0 = oracle.grad(start_point)
    grad0_norm = np.linalg.norm(grad0)**2
    
    w = start_point
    for i in range(1, max_iter):
        fi, grad = oracle.fuse_value_grad(w)
        direction = grad.reshape(-1, 1)

        assert(w.shape == direction.shape), "diff shape for grad and w"
        
        alpha = line_search(oracle, w, direction)
        w = w - alpha * direction
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2 / grad0_norm}")
        
        if np.linalg.norm(grad)**2 / grad0_norm <= tol:
            break

    return w


def optimize_newton(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000, verbose=True):
    line_search = getattr(step_search, f"{line_search_method}_line_search")
 
    w = start_point 
    for i in range(1, max_iter):
        fi, grad, hessian = oracle.fuse_value_grad_hessian(w)
        
        pos_hessian = shift_positive_definite(hessian)
        direction = solve_cholesky(pos_hessian, grad.reshape(-1, 1))
        # direction = (grad @ np.linalg.inv(pos_hessian)).reshape(-1, 1)    
    
        assert(w.shape == direction.shape), "diff shape for grad and w"
        
        alpha = line_search(oracle, w, direction)
        w = w - alpha * direction
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2}")
        
        if np.linalg.norm(grad)**2 <= tol:
            break

    return w


def optimize_hfn(oracle, start_point, line_search_method, tol=1e-8, max_iter=10000, verbose=True):
    line_search = getattr(step_search, f"{line_search_method}_line_search")
 
    w, pred_fi = start_point, float("inf")
    for i in range(1, max_iter):
        fi, grad = oracle.fuse_value_grad(w)
        
        # H * p = grad
        direction = conjugate_grad(lambda d: oracle.hessian_vec_product(w=w, d=d), grad.reshape(-1, 1))
    
        assert(w.shape == direction.shape), "diff shape for grad and w"
        
        alpha = line_search(oracle, w, direction)
        w = w - alpha * direction
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2}")
        
        if np.linalg.norm(grad)**2 <= tol or abs(pred_fi - fi) <= 1e-12:
            break
        
        pred_fi = fi

    return w    

        
def main():
    oracle = make_oracle("data/a1a.txt")
    # oracle = make_oracle()

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    # w_init = np.random.normal(size=w_n).reshape(-1, 1)
    # w_init = np.ones(w_n).reshape(-1, 1)

    optimize_hfn(oracle, w_init, line_search_method="brent")

    
if __name__ == "__main__":
    main()
    
