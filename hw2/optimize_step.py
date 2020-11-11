import scipy
import numpy as np
import step_search

from oracle import make_oracle
from utils import shift_positive_definite, solve_cholesky

    

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

    assert(w.shape == direction.shape), "diff shape for grad and w"
    
    alpha = line_search(oracle, w, direction)
    w = w - alpha * direction
    
    grad_ratio = np.linalg.norm(grad)**2 / grad0_norm
    
    if verbose and iter_num % 1 == 0:
        print(f"Iteration {iter_num}: {fi}, alpha: {alpha}, grads: {grad_ratio}")
    
    return w, (grad_ratio <= tol)

