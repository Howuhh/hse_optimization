import numpy as np

from optimizers import golden_section_search, ternary_search
from optimizers import inverse_parabola_interpolation, brent


# TODO: add verbose
def optimize(f, a, b, eps=1e-8, max_iter=200, method="brent", true_min=None):
    if method == "golden":
        optimizer = golden_section_search
    elif method == "ternary":
        optimizer = ternary_search
    elif method == "parabola":
        optimizer = inverse_parabola_interpolation
    elif method == "brent":
        optimizer = brent
    else:
        raise ValueError("Unknown method")

    return optimizer(f, a, b, eps, max_iter, true_min)