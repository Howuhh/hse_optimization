import numpy as np

from scipy.sparse import csr_matrix


def generate_dataset(n=50, w_dim=1, sparse=False):
    X = np.random.normal(size=(n, w_dim))
    X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
    
    w = np.random.uniform(-1, 1, size=(w_dim + 1, 1)).astype(np.float64)
    
    y = (X @ w >= 0).astype(np.int)
    
    if sparse:
        X = csr_matrix(X)

    return X, y, w


def confusion_matrix(true, pred):
    result = np.zeros((2, 2))

    for i in range(len(true)):
        result[int(pred[i]), int(true[i])] += 1

    return result


def rell_error(pred, true_):
    return (abs(pred - true_) / true_).max()


def multiply(matrix, const):
    if isinstance(matrix, np.ndarray):
        return matrix * const
    else:
        return matrix.multiply(const)


def shift_positive_definite(X):
    alpha_I = 1e-6 * np.identity(X.shape[0])
    
    while np.any(np.linalg.eigvals(X) <= 0):
        X = X + alpha_I

    return X


def solve_cholesky(A, b):
    L = np.linalg.cholesky(A)
    
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(np.matrix(L).H, y)
    
    return x


def inexact_conjugate_grad(hess_vec_prod, grad, max_iter=1000):
    x = np.zeros_like(grad)
    r, direction = grad, -grad
    
    grad_norm = np.linalg.norm(grad)
    eps = min(0.3, np.sqrt(grad_norm)) * grad_norm
    
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


if __name__ == "__main__":
    print(generate_dataset(5))