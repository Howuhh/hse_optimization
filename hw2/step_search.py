import numpy as np

from scipy.optimize import bracket

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