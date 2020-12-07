import utils
import numpy as np

from scipy.special import expit


def binary_cross_entropy(X, y, w):
    N, Xw = X.shape[0], X @ w
 
    loss = np.sum(y * np.log(expit(Xw) + 1e-12) + (1 - y) * np.log(1 - expit(Xw) + 1e-12))
    
    return -1/N * loss


def entropy_grad(X, y, w):
    N = X.shape[0]
    
    return (expit(X @ w) - y).T @ X / N


def entropy_hessian(X, w):
    N, Xw = X.shape[0], X @ w

    hessian = X.T @ utils.multiply(X, expit(Xw) * (1 - expit(Xw))) / N
    
    if hasattr(hessian, "toarray"):
        return hessian.toarray()
    
    return hessian


def entropy_hessian_product(X, w, d):
    sigma = expit(X @ w)
    sigma = sigma * (1 - sigma)
    step1 = X @ d
    step2 = sigma.reshape(-1, 1) * step1
    step3 = X.T @ step2
    return step3 / X.shape[0]