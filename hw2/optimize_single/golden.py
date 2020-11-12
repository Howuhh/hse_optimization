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


# def golden_section(f, a, b, tol=1e-5, max_iter=25):
#     invphi = (np.sqrt(5) - 1) * 0.5 
#     invphi2 = (3 - np.sqrt(5)) * 0.5 
    
#     (a, b) = (min(a, b), max(a, b))
#     h = b - a
#     if h <= tol:
#         return (a, b)

#     c = a + invphi2 * h
#     d = a + invphi * h
#     yc = f(c)
#     yd = f(d)

#     for k in range(max_iter):
#         if yc < yd:
#             b = d
#             d = c
#             yd = yc
#             h = invphi * h
#             c = a + invphi2 * h
#             yc = f(c)
#         else:
#             a = c
#             c = d
#             yc = yd
#             h = invphi * h
#             d = a + invphi * h
#             yd = f(d)

#     if yc < yd:
#         return (a + d) / 2
#     else:
#         return (c + b) / 2