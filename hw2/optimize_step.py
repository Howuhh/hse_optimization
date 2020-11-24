import scipy
import numpy as np

# from utils import shift_positive_definite, solve_cholesky
from utils import shift_positive_definite_cho, inexact_conjugate_grad

    
def descent_step(oracle, w, grad0_norm, line_search, tol):
    fi, grad = oracle.fuse_value_grad(w)
    direction = grad.reshape(-1, 1)

    alpha = line_search(oracle, w, direction)
    alpha = 1.0
    w = w - alpha * direction
    
    grad_ratio = np.linalg.norm(grad)**2 / grad0_norm
    
    return w, (grad_ratio <= tol), (fi, alpha, grad_ratio)


def newton_step(oracle, w, grad0_norm, line_search, tol):
    fi, grad, hessian = oracle.fuse_value_grad_hessian(w)

    L = shift_positive_definite_cho(hessian)
    direction = scipy.linalg.cho_solve((L, True), grad.reshape(-1, 1))
    
    alpha = line_search(oracle, w, direction)
    # if np.allclose(alpha, 0.0):
        # alpha = 1.0
    # w = w - alpha * direction / np.linalg.norm(direction)
    w = w - alpha * direction
    
    grad_norm = np.linalg.norm(grad)**2 / grad0_norm
    
    return w, (grad_norm <= tol), (fi, alpha, grad_norm)


def hf_newton_step(oracle, w, grad0_norm, line_search, tol, cg_tol):
    fi, grad = oracle.fuse_value_grad(w)

    # H @ p = -grad
    direction = inexact_conjugate_grad(
        lambda d: oracle.hessian_vec_product(w=w, d=d), 
        grad=grad.reshape(-1, 1), 
        tol=cg_tol
    )
        
    alpha = line_search(oracle, w, -direction)
    # alpha = line_search(oracle, w, -direction)
    
    # if np.max(np.abs(grad)) > 0.1:
        # direction = direction / np.linalg.norm(direction)
    # w = w + alpha * (direction / np.linalg.norm(direction)) # may stuck in place without normalization
    w = w + alpha * direction

    grad_norm = np.linalg.norm(grad)**2 / grad0_norm

    return w, (grad_norm <= tol), (fi, alpha, grad_norm)