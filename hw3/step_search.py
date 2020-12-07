import warnings
import numpy as np

from scipy.optimize import bracket, line_search
from scipy.optimize.linesearch import LineSearchWarning


def armijo_line_search(oracle, w, direction):    
    def f(alpha):
        return oracle.value(w - alpha * direction)
    
    alpha = np.mean(bracket(f)[:3])
    c = 1e-4 # 0.0001
    
    fk = oracle.value(w)
    grad_norm = oracle.grad(w) @ direction

    i = 0
    while oracle.value(w - alpha * direction) > fk + alpha * c * grad_norm and i < 10000:
        alpha = alpha / 2
        i += 1
        
    return alpha


def wolfe_line_search(oracle, w, direction, not_converge="armijo"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=LineSearchWarning)
        alpha = line_search(oracle.value, oracle.grad, w, -direction, direction.T)[0]
    
    if alpha is None:
        if not_converge == "armijo":
            alpha = armijo_line_search(oracle, w, direction)
        else:
            alpha = 1.0

    return alpha