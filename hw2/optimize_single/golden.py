import numpy as np
from scipy.optimize import bracket


# TODO: optimize golden section
def golden_section_navie(f, a, b, eps=1e-5, max_iter=25):
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


def golden_section(f, brack, eps=1e-5, max_iter=25):
    ax, cx, bx = brack
    
    x0, x3 = ax, cx
    phi = (np.sqrt(5) - 1) / 2
    phi_inv = 1 - phi
    
    if abs(cx - bx) > abs(bx - ax):
        x1 = bx
        x2 = bx + phi_inv * (cx - bx)
    else:
        x2 = bx
        x1 = bx - phi_inv * (bx - ax)
        
    f1, f2 = f(x1), f(x2)
    
    for i in range(max_iter):
        if abs(x3 - x0) < eps * (abs(x1) + abs(x2)):
            break
        if f2 < f1:
            x0, x1, x2 = x1, x2, phi * x2 + phi_inv * x3
            f1, f2 = f2, f(x2)
        else:
            x3, x2, x1 = x2, x1, phi * x1 + phi_inv * x0
            f2, f1 = f1, f(x1)
    
    return x1 if f1 < f2 else x2


if __name__ == "__main__":
    f = lambda x: x**2
    a, b, c = bracket(f, -5, 5)[:3]
    
    print(round(golden_section(f, (a, c, b)), 3))
    
    
    