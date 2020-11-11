import numpy as np
import step_search

from time import time
from oracle import make_oracle
from optimize_log import OptimizeLog

from optimize_step import descent_step, newton_step, hf_newton_step
    

def _optimize(optimize_step, oracle, w, line_search_method, tol, max_iter, verbose):
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
        fi, alpha, grad_norm = info
        log.add_log(time() - log.start_time, fi, alpha, grad_norm)
        
        if verbose and i % 1 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {grad_norm}")
        
        if stop_condition:
            break
        
    return w, log


def optimize_gd(oracle, start_point, line_search_method="brent", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(descent_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def optimize_newton(oracle, start_point, line_search_method="wolfe", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose)


def optimize_hfn(oracle, start_point, line_search_method="armijo", tol=1e-8, max_iter=10000, verbose=False):
    return _optimize(hf_newton_step, oracle, start_point, line_search_method, tol, max_iter, verbose)
    




# def optimize_gd(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000, verbose=True):
#     line_search = getattr(step_search, f"{line_search_method}_line_search")
    
#     grad0 = oracle.grad(start_point)
#     grad0_norm = np.linalg.norm(grad0)**2
    
#     w = start_point
#     for i in range(1, max_iter + 1):
#         fi, grad = oracle.fuse_value_grad(w)
#         direction = grad.reshape(-1, 1)

#         assert(w.shape == direction.shape), "diff shape for grad and w"
        
#         alpha = line_search(oracle, w, direction)
#         w = w - alpha * direction
        
#         if verbose and i % 100 == 0:
#             print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2 / grad0_norm}")
        
#         if np.linalg.norm(grad)**2 / grad0_norm <= tol:
#             break

#     return w


# def optimize_newton(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000, verbose=True):
#     line_search = getattr(step_search, f"{line_search_method}_line_search")
 
#     w = start_point 
#     for i in range(1, max_iter + 1):
#         fi, grad, hessian = oracle.fuse_value_grad_hessian(w)
        
#         pos_hessian = shift_positive_definite(hessian)
#         direction = solve_cholesky(pos_hessian, grad.reshape(-1, 1))
#         # direction = (grad @ np.linalg.inv(pos_hessian)).reshape(-1, 1)    
    
#         assert(w.shape == direction.shape), "diff shape for grad and w"
        
#         alpha = line_search(oracle, w, direction)
#         w = w - alpha * direction
        
#         if verbose and i % 1 == 0:
#             print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2}")
        
#         if np.linalg.norm(grad)**2 <= tol:
#             break

#     return w


# def optimize_hfn(oracle, start_point, line_search_method="wolfe", tol=1e-8, max_iter=10000, verbose=True):
#     line_search = getattr(step_search, f"{line_search_method}_line_search")
 
#     w = start_point
#     for i in range(1, max_iter + 1):
#         fi, grad = oracle.fuse_value_grad(w)

#         # H * p = grad
#         direction = inexact_conjugate_grad(
#             lambda d: oracle.hessian_vec_product(w=w, d=d), 
#             grad=grad.reshape(-1, 1)
#         )

#         alpha = line_search(oracle, w, -direction)
#         w = w + alpha * (direction / np.linalg.norm(direction)) # may stuck in place without normalization
        
#         if verbose and i % 1 == 0:
#             print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2}")
        
#         if np.linalg.norm(grad)**2 <= tol:
#             break
        
#     return w    

        
def main():
    oracle = make_oracle("data/a1a.txt")
    # oracle = make_oracle()

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    # w_init = np.random.uniform(size=w_n).reshape(-1, 1)
    # w_init = np.random.normal(size=w_n).reshape(-1, 1)
    # w_init = np.ones(w_n).reshape(-1, 1)

    optimize_gd(oracle, w_init, "lipschitz")

    
if __name__ == "__main__":
    main()
    
