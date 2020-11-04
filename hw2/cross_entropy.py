import numpy as np

from math import inf

from utils import multiply, generate_dataset, rell_error
from grad_check import eval_num_grad, eval_num_hessian

from scipy.special import expit

def sigmoid(z, safe=False):
    assert isinstance(z, np.ndarray)

    # return np.exp(z) / (1 + np.exp(z))
    # much faster and stable (no overflow)
    return expit(z)    


def binary_cross_entropy(X, y, w):
    N = X.shape[0]
    
    loss = np.sum(y * np.log(sigmoid(X @ w) + 1e-12) + (1 - y) * np.log(1 - sigmoid(X @ w) + 1e-12))
    return -1/N * loss


def entropy_grad(X, y, w):
    N = X.shape[0]
    
    return (sigmoid(X @ w) - y).T @ X / N


def entropy_hessian(X, w):
    N = X.shape[0]
    
    hessian = X.T @ multiply(X, sigmoid(X @ w) * (1 - sigmoid(X @ w))) / N
    
    if hasattr(hessian, "toarray"):
        return hessian.toarray()
    
    return hessian
 

def test_grad(n=1000, iters=1000):
    grad_err, hess_err = [], []
    
    for _ in range(1000):
        X, y, w = generate_dataset(n=1000)
        
        num_grad = eval_num_grad(lambda w: binary_cross_entropy(X, y, w=w.reshape(-1, 1)), w.reshape(1, -1)[0])
        true_grad = entropy_grad(X, y, w)
        
        grad_err.append(rell_error(true_grad, num_grad))
        
        num_hessian = eval_num_hessian(lambda w: binary_cross_entropy(X, y, w=w.reshape(-1, 1)), w.reshape(1, -1)[0])
        true_hessian = entropy_hessian(X, w)
        
        hess_err.append(rell_error(true_hessian, num_hessian))

    print(f"Mean relative error (n={n}, iter={iters})")
    print("--"*14)
    print("gradient: ", np.mean(grad_err))
    print("hessian: ", np.mean(hess_err))



if __name__ == "__main__":
    test_grad()
    