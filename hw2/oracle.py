import scipy
import numpy as np

from utils import generate_dataset

from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer

from cross_entropy import binary_cross_entropy, entropy_grad, entropy_hessian


class Oracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.L = 0.4
        
        self._h = np.power(1.1 * 10e-16, 1/3)
        self._call_count = 0

    def value(self, w):       
        self._call_count += 1 
        return binary_cross_entropy(self.X, self.y, w)

    def grad(self, w):
        self._call_count += 1        
        return entropy_grad(self.X, self.y, w)

    def hessian(self, w):
        self._call_count +=1
        return entropy_hessian(self.X, w)

    def hessian_vec_product(self, w, d):
        self._call_count += 1   
        f_f = self.grad(w + self._h * d).reshape(-1, 1)
        f_b = self.grad(w - self._h * d).reshape(-1, 1)
        
        return (f_f - f_b) / (2*self._h)

    def fuse_value_grad(self, w):
        self._call_count -= 1
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        self._call_count -= 2
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        self._call_count -= 2
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)


def make_oracle(data_path=None, sparse=False):
    if data_path is None:
        X, y, _ = generate_dataset(n=1000, w_dim=3)
    else:
        X, y = load_svmlight_file(data_path)
        
        X = scipy.sparse.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
        y = LabelBinarizer().fit_transform(y)
    
    return Oracle(X, y)


if __name__ == "__main__":
    oracle = make_oracle()
    w = np.random.uniform(size=(oracle.X.shape[1], 1))
    
    assert np.allclose(oracle.hessian(w) @ w, oracle.hessian_vec_product(w, w))