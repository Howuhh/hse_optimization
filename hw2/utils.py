import oracle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix


def generate_dataset(n=50, w_dim=1, sparse=False):
    X = np.random.normal(size=(n, w_dim))
    X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
    
    w = np.random.uniform(-1, 1, size=(w_dim + 1, 1)).astype(np.float64)
    
    y = (X @ w >= 0).astype(np.int)
    
    if sparse:
        X = csr_matrix(X)

    return X, y, w


def rell_error(pred, true_):
    return (abs(pred - true_) / true_).max()


def multiply(matrix, const):
    if isinstance(matrix, np.ndarray):
        return matrix * const
    else:
        return matrix.multiply(const)


def shift_positive_definite_cho(X):
    eps, I = 1e-16, np.identity(X.shape[0])

    while True:
        try:
            L = np.linalg.cholesky(X)
            break
        except np.linalg.LinAlgError:
            X = X + eps * I
            eps = eps * 2
              
    return L


def inexact_conjugate_grad(hess_vec_prod, grad, tol, max_iter=1000):
    x = np.zeros_like(grad)
    r, direction = grad, -grad
    
    grad_norm = np.linalg.norm(grad)
    
    if tol == "sqrt":
        eps = min(0.1, np.sqrt(grad_norm)) * grad_norm
    elif tol == "norm":
        eps = min(0.1, grad_norm) * grad_norm
    else:
        eps = 1e-4
    
    for i in range(max_iter):
        Hd = hess_vec_prod(direction)
        
        pos_def = direction.T @ Hd
        grad_dot_old = r.T @ r
        
        if pos_def <= 0:
            if i == 0:
                return -grad
            else:
                return x
                        
        alpha = grad_dot_old / pos_def
        x = x + alpha * direction
        r = r + alpha * Hd
        
        if np.linalg.norm(r) < eps:
            return x
        
        beta = r.T @ r / grad_dot_old
        direction = -r + beta * direction

    return x


def run_optimizer(data_path, optimizer, **kwargs):
    oracle_ = oracle.make_oracle(data_path)
    w_init = np.zeros(oracle_.X.shape[1]).reshape(-1, 1)
    
    w, log = optimizer(oracle_, w_init, **kwargs)
    return log


def plot_metric(log, method, xaxis, offset=0):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    for line_search in log[method]:
        line_search_log = log[method][line_search]
        info = line_search_log.get_log()
        
        error = np.log10(line_search_log.error)[offset:]
        grads = np.log10(info["grad_info"])[offset:]
        
        if xaxis == "num_iter":
            metric = np.arange(error.shape[0])
        else:
            metric = info[xaxis][offset:]
        
        ax[0].plot(metric, error)
        ax[1].plot(metric, grads, label=f"{line_search}: {round(info['entropy'][-1],6)}")
        
    ax[0].set(title="Convergence by $|F(w^*) - F(w)|$", 
              xlabel=xaxis, ylabel="$\\log_{10} |F(w^*) - F(w)|$")
    ax[1].set(title="Convergence by $\\frac{|\\nabla F(w)|}{|\\nabla F(w_0)|}$",
              xlabel=xaxis, ylabel="$\\log_{10} \\frac{|\\nabla F(w)|}{|\\nabla F(w_0)|}$")
    
    ax[1].legend()


if __name__ == "__main__":
    print(generate_dataset(5))