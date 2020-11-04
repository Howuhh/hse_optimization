import numpy as np


def parabola_min(x1, x2, x3, fx1, fx2, fx3):
    x2sx1 = x2 - x1
    x2sx3 = x2 - x3
    f2sf1 = fx2 - fx1
    f2sf3 = fx2 - fx3
    
    u = x2 - (x2sx1**2*f2sf3 - x2sx3**2*f2sf1) / (2*(x2sx1*f2sf3 - x2sx3*f2sf1) + 1e-8)
    return u


def optimize(f, a, b, eps=1e-8, max_iter=200):
    K = (3 - 5 ** 0.5) / 2
    x = w = v = a + K * (b - a)
    fx = fw = fv = f(x)[0]

    d = e = b - a
    for iter_ in range(max_iter):
        g, e = e, d
        tol = eps * abs(x) + eps

        # convergence condition
        if abs(x - (a + b) * 0.5) + (b - a) * 0.5 <= 2*tol:
            break

        parabola_u = False
        # parabola interpolation through x, w, v (all different)
        if (x != w != v) and (fx != fw != fv):
            # should be x1 < x2 < x3 
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

        fu = f(u)[0]
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

    return x


if __name__ == "__main__":
    f = lambda x: (np.sin(x), 1)

    print(round(optimize(f, -3, 3), 4))