import numpy as np

from scipy.optimize import bracket, line_search

from optimize_single.golden import golden_section
from optimize_single.brent import brent


def _line_search(oracle, w, direction, optimizer, eps, max_iter):
    assert w.shape == direction.shape, "diff shape for w and direction"
    
    def f(alpha):
        return oracle.value(w - alpha * direction)

    xa, xb = 1, 10  # some magick numbers for which Brent works well
    if f(xa) > f(xb):
        xa, xb = xb, xa

    brack = bracket(f, xa=xa, xb=xb)[:3]

    return optimizer(f, brack, eps=eps, max_iter=max_iter)
    
    
def golden_line_search(oracle, w, direction):
    # golden really expensive method, so only few iterations
    return _line_search(oracle, w, direction, golden_section, 1e-5, 16)  


def brent_line_search(oracle, w, direction):
    return _line_search(oracle, w, direction, brent, 1e-8, 64) 


def armijo_line_search(oracle, w, direction):    
    def f(alpha):
        return oracle.value(w - alpha * direction)
    
    alpha = max(bracket(f)[:3])
    p, c = 0.5, 0.0001
    
    fk = oracle.value(w)
    grad_norm = c * oracle.grad(w) @ direction

    i = 0
    while oracle.value(w - alpha * direction) >= fk + alpha * grad_norm and i <= 10000:
        alpha = p * alpha
        i += 1
        
    return alpha


def wolfe_line_search(oracle, w, direction):
    alpha = line_search(oracle.value, oracle.grad, w, -direction, direction.T)[0]
    
    if alpha is None:
        alpha = armijo_line_search(oracle, w, direction)

    return alpha


def lipschitz_line_search(oracle, w, direction):
    L = oracle.L
    
    w_new = w - (1 / L) * direction

    fw = oracle.value(w)
    p_norm = np.linalg.norm(direction)**2
    grad_dot_d = direction.T @ direction

    while oracle.value(w_new) > fw + (1 / L) * grad_dot_d + (1 / (2*L)) * p_norm:
        L = L * 2
        w_new = w_new - (1 / L) * direction
        # w_new = w - (1 / L) * direction  # not working that way

    oracle.L = max(L / 2, 0.4)

    return 1 / L