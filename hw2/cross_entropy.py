import numpy as np

from utils import multiply, generate_dataset, rell_error
from num_grad import eval_num_grad, eval_num_hessian

from scipy.special import expit


def sigmoid(z):
    assert isinstance(z, np.ndarray)

    return expit(z)    


def binary_cross_entropy(X, y, w):
    N, Xw = X.shape[0], X @ w

    loss = np.sum(y * np.log(sigmoid(Xw) + 1e-15) + (1 - y) * np.log(1 - sigmoid(Xw) + 1e-15))
    # loss = y.T @ np.log(sigmoid(X @ w) + 1e-12) + (1 - y).T @ np.log(1 - sigmoid(X @ w) + 1e-12)
    
    return -1/N * loss


def entropy_grad(X, y, w):
    N = X.shape[0]
    
    return (sigmoid(X @ w) - y).T @ X / N


def entropy_hessian(X, w):
    N, Xw = X.shape[0], X @ w
    
    hessian = X.T @ multiply(X, sigmoid(Xw) * (1 - sigmoid(Xw))) / N
    
    if hasattr(hessian, "toarray"):
        return hessian.toarray()
    
    return hessian
 

def test_grad(n=1000, iters=1000):
    grad_err, hess_err = [], []
    
    for _ in range(1000):
        X, y, _ = generate_dataset(n=1000)
        w_init = np.random.uniform(size=(X.shape[1], 1))
                
        num_grad = eval_num_grad(lambda w: binary_cross_entropy(X, y, w=w.reshape(-1, 1)), w_init.ravel())
        true_grad = entropy_grad(X, y, w_init)
        
        grad_err.append(rell_error(true_grad, num_grad))
        
        num_hessian = eval_num_hessian(lambda w: binary_cross_entropy(X, y, w=w.reshape(-1, 1)), w_init.ravel())
        true_hessian = entropy_hessian(X, w_init)
        
        hess_err.append(rell_error(true_hessian, num_hessian))

    print(f"Mean relative error (n={n}, iter={iters})")
    print("--"*14)
    print("gradient: ", np.mean(grad_err))
    print("hessian: ", np.mean(hess_err))


if __name__ == "__main__":
    test_grad()
    