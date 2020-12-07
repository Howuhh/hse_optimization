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


def armijo_line_search(oracle, w, direction, init_alpha="mean"):    
    def f(alpha):
        return oracle.value(w - alpha * direction)
    
    brack = bracket(f)
    x, fx = np.array(brack[:3]), np.array(brack[3:-1])
    
    if init_alpha == "wmean":
        alpha = np.mean(x * fx)  # magick trick to boost armijo from 7k iterations to 2k iterations on gd
    elif init_alpha == "mean":
        alpha = np.mean(x) # a little less magick trick to boost armijo to 3.5k without breaking on other (not a1a) datasets 
    elif init_alpha == "max":
        alpha = max(x)
    else:
        alpha = 100

    c = 0.0001
    
    fk = oracle.value(w)
    grad_norm = oracle.grad(w) @ direction

    i = 0
    while oracle.value(w - alpha * direction) > fk + alpha * c * grad_norm and i < 10000:
        alpha = alpha / 2
        i += 1
        
    return alpha


def wolfe_line_search(oracle, w, direction):
    # TODO: not direction.T but gradient to wolfe
    alpha = line_search(oracle.value, oracle.grad, w, -direction, direction.T)[0]
    
    if alpha is None:
        alpha = armijo_line_search(oracle, w, direction, init_alpha="max")

    return alpha


def lipschitz_line_search(oracle, w, direction):
    L = oracle.L # init with 1.0

    dir_norm = 0.5 * direction.T @ direction

    w_new = w - (1/L) * direction
    while oracle.value(w_new) > oracle.value(w) - (1 / L) * dir_norm:
        L = L * 2
        w_new = w - (1/L) * direction
        
    oracle.L = (L / 2)
    
    return 1 / oracle.L