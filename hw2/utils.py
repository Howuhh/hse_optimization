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


def conjugate_grad(hess_vec_prod, b):
    x = np.ones(b.shape[1]).reshape(-1, 1)
    
    grad = hess_vec_prod(x) - b # A @ x - b
    direction = -grad
    
    while np.linalg.norm(grad)**2 > 1e-8:
        grad_dot_old = grad.T @ grad
        A_p = hess_vec_prod(direction) # A @ direction
        
        alpha = grad_dot_old / (direction.T @ A_p)
        
        x = x + alpha * direction
        grad = grad + alpha * A_p
        
        beta = grad.T @ grad / grad_dot_old
        direction = - grad + beta * direction

    return x


if __name__ == "__main__":
    print(generate_dataset(5))