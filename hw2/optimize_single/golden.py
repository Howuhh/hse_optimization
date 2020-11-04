import numpy as np


# TODO: optimize golden section
def golden_section(f, a, b, eps=1e-5, max_iter=25):
    phi = (1 + np.sqrt(5)) / 2
    
    iter_ = 0
    while abs(a - b) >= eps and iter_ <= max_iter:
        x1 = b - ((b - a) / phi)
        x2 = a + ((b - a) / phi)

        y1, y2 = f(x1), f(x2)
       
        if y1 >= y2:
            a = x1
        else:
            b = x2

    return (a + b) / 2