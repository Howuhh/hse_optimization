import numpy as np


def eval_num_grad(f, w, h=None):
    "central difference gradient approximation"
    if h is None:
        # from book on numerical optimization springer
        h = np.power(1.1 * 10e-16, 1/3)
    
    grad = np.zeros_like(w, dtype=np.float64)
    
    e = h * np.identity(w.shape[0])
    for i in range(w.shape[0]):
        f_left = f(w + e[i, :])
        f_right = f(w - e[i, :])

        grad[i] = (f_left - f_right) / (2*h)
        
    return grad


def eval_num_hessian(f, w, h=None):
    "forward difference hessian approximation"
    if h is None:
        h = np.power(1.1 * 10e-16, 1/3)
    n = w.shape[0]
    
    hessian = np.zeros((n, n), dtype=np.float64)
    e = h * np.identity(n)
    
    fw = f(w)
    fws = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        fws[i] = f(w + e[i, :])
    
    for i in range(n):
        for j in range(i, n):
            fw_ij = f(w + e[i, :] + e[j, :])
            
            hessian[i, j] = (fw_ij - fws[i] - fws[j] + fw) / (h*h)
            hessian[j, i] = hessian[i, j]
    
    return hessian


if __name__ == "__main__":
    f = lambda w: np.sum(w) / w.shape[0]
    
    assert np.allclose(eval_num_grad(f, np.array([1, 2])), [0.5, 0.5]), "eval_num_grad does not converge!"
    assert np.allclose(eval_num_hessian(f, np.array([1, 2])), np.array([0, 0, 0, 0]).reshape(2, 2)), "eval_num_hessian does not converge!"
    