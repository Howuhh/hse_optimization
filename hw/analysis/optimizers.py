from time import time
import numpy as np

from math import inf

from dataclasses import dataclass
from dataclasses import field
from typing import List


@dataclass
class OptimizeLog:
    start_time: float
    err: List[float] = field(default_factory=list)
    iters: List[int] = field(default_factory=list)
    time: List[float] = field(default_factory=list)
    iter_min: List[float] = field(default_factory=list)


def parabola_min(x1, x2, x3, fx1, fx2, fx3):
    x2sx1 = x2 - x1
    x2sx3 = x2 - x3
    f2sf1 = fx2 - fx1
    f2sf3 = fx2 - fx3
    
    u = x2 - (x2sx1**2*f2sf3 - x2sx3**2*f2sf1) / (2*(x2sx1*f2sf3 - x2sx3*f2sf1) + 1e-12)
    return u


def brent(f, a, b, eps, max_iter, true_min=None):
    if true_min is not None:
        log = OptimizeLog(start_time=time())

    K = (3 - 5 ** 0.5) / 2
    x = w = v = a + K * (b - a)
    fx = fw = fv = f(x)

    d = e = b - a
    for iter_ in range(int(max_iter)):
        g, e = e, d
        # tol = eps * abs(x) + eps / 10
        tol = eps * abs(x) + eps
        
        # convergence condition
        if abs(x - (a + b) * 0.5) <= 2*tol - 0.5 * (b - a):
            break

        parabola_u = False
        # parabola interpolation through x, w, v (all different)
        if (x != w != v) and (fx != fw != fv):
            # should be x1 < x2 < x3 
            # ideally fx is min, so only v < x < w or w < x < v is possible, but... 
            if x < w:
                if w < v:
                    # x < w < v
                    u = parabola_min(x, w, v, fx, fw, fv)
                elif x < v:
                    # x < v < w
                    u = parabola_min(x, v, w, fx, fv, fw)
                else:
                    # v < x < w
                    u = parabola_min(v, x, w, fv, fx, fw)
            else:
                # x > w
                if x < v:
                    # w < x < v
                    u = parabola_min(w, x, v, fw, fx, fv)
                elif w < v:
                    # w < v < x
                    u = parabola_min(w, v, x, fw, fv, fx)
                else:
                    # v < w < x
                    u = parabola_min(v, w, x, fv, fw, fx)            
            
            if a <= u <= b and abs(x - u) < g/2:
                parabola_u = True
                if u - a < 2 * tol and b - u < 2 * tol:
                    u = x - np.sign(x - (a + b) / 2) * tol
        
        # golden section method
        if parabola_u is False:
            if x < (a + b) / 2:
                u = x + K * (b - x)  # [x, b]
                e = b - x
            else:
                u = x - K * (x - a)  # [a, x]
                e = x - a 
        
        # min interval len
        if abs(u - x) < tol:
            u = x + np.sign(u - x) * tol
        d = abs(u - x)

        fu = f(u)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u >= x:
                b = u
            else:
                a = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v = u
                fv = fu

        if true_min is not None:
            log.time.append(time() - log.start_time)
            log.iters.append(iter_)
            log.err.append(abs(true_min - x))
            log.iter_min.append([x, fx])

    if true_min is not None:
        return x, log

    return x


def inverse_parabola_interpolation(f, a, b, eps, max_iter, true_min=None):
    if true_min is not None:
        log = OptimizeLog(start_time=time())

    iter_, prev_u = 0, inf

    x2 = (a + b) / 2
    fa, fx2, fb = f(a), f(x2), f(b)
    
    while iter_ <= max_iter:    
        u = parabola_min(a, x2, b, fa, fx2, fb)

        fu = f(u)
        # x2 to the left of u or to the right
        # new x2 should be between new a & b
        if x2 < u:
            if fx2 >= fu:
                a = x2
                fa = fx2

                x2 = u
                fx2 = fu
            else:
                b = u
                fb = fu
        else:
            if fx2 >= fu:
                b = x2
                fb = fx2
                
                x2 = u
                fx2 = fu
            else:
                a = u
                fa = fu
                
        if abs(prev_u - u) <= eps:
            break

        if true_min is not None:
            log.time.append(time() - log.start_time)
            log.iters.append(iter_)
            log.err.append(abs(true_min - u))

        iter_ += 1
        prev_u = u

    if true_min is not None:
        return u, log

    return u


def golden_section_search(f, a, b, eps, max_iter, true_min=None):
    if true_min is not None:
        log = OptimizeLog(start_time=time())

    phi = (1 + np.sqrt(5)) / 2

    iter_ = 0
    while abs(a - b) >= eps and iter_ <= max_iter:
        x_1 = b - ((b - a) / phi)
        x_2 = a + ((b - a) / phi)
        y_1, y_2 = f(x_1), f(x_2)
       
        if y_1 >= y_2:
            a = x_1
        else:
            b = x_2

        if true_min is not None:
            log.time.append(time() - log.start_time)
            log.iters.append(iter_)
            log.err.append(abs(true_min - (a + b) / 2))
            log.iter_min.append([(a + b) / 2, f((a + b) / 2)])

        iter_ += 1

    if true_min is not None:
        return (a + b) / 2, log

    return (a + b) / 2


def ternary_search(f, a, b, eps, max_iter, true_min=None):
    if true_min is not None:
        log = OptimizeLog(start_time=time())

    iter_ = 0
    while abs(a - b) >= eps and iter_ <= max_iter:
        x_1 = a + ((b - a) / 3)
        x_2 = b - ((b - a) / 3)
        y_1, y_2 = f(x_1), f(x_2)

        if y_1 >= y_2:
            a = x_1
        else:
            b = x_2
        iter_ += 1

        if true_min is not None:
            log.time.append(time() - log.start_time)
            log.iters.append(iter_)
            log.err.append(abs(true_min - ((a + b) / 2)))

    if true_min is not None:
        return (a + b) / 2, log
        
    return (a + b) / 2

    