import scipy
import numpy as np

from scipy.optimize import bracket, line_search

from optimize_single.golden import golden_section
from optimize_single.brent import brent


def _line_search(oracle, w, direction, optimizer):
    assert w.shape == direction.shape, "diff shape for w and direction"
    
    def f(alpha):
        return oracle.value(w - alpha * direction)
    
    xa, xb = 1, 15
    if f(xa) > f(xb):
        xa, xb = xb, xa
        
    a, c, b = bracket(f, xa=xa, xb=xb)[:3]
        
    return optimizer(f, a, b, 1e-5, 50)
    
    
def golden_line_search(oracle, w, direction):
    return _line_search(oracle, w, direction, golden_section) 


def brent_line_search(oracle, w, direction):        
    return _line_search(oracle, w, direction, brent) 


def armijo_line_search(oracle, w, direction):    
    alpha, p, c = 5, 0.5, 0.0001
    
    fk = oracle.value(w)
    grad_norm = c * oracle.grad(w) @ direction

    i = 0
    while oracle.value(w - alpha * direction) >= fk + alpha * grad_norm and i <= 100:
        alpha = p * alpha
        i += 1
        
    return alpha


def wolfe_line_search(oracle, w, direction):
    alpha = line_search(oracle.value, oracle.grad, w, -direction, direction.T)[0]
    
    if alpha is None:
        alpha = armijo_line_search(oracle, w, direction)

    return alpha


def lipschitz_line_search(oracle, w, direction):
    L = 1.0 
    w_new = w - (1 / L) * direction

    fw = oracle.value(w)

    while oracle.value(w_new) > fw + (w_new - w).T @ direction + (L / 2) * np.linalg.norm(w_new - w)**2:
        L = L * 2
        w_new = w_new - (1 / L) * direction

    L = max(1.0, L / 2)

    return 1 / L