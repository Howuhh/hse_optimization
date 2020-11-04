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
        
    def value(self, w):
        return binary_cross_entropy(self.X, self.y, w)

    def grad(self, w):
        return entropy_grad(self.X, self.y, w)

    def hessian(self, w):
        return entropy_hessian(self.X, w)

    def hessian_vec_product(self, w, d):
        pass

    def fuse_value_grad(self, w):
        return self.value(w), self.grad(w)

    def fuse_value_grad_hessian(self, w):
        return self.value(w), self.grad(w), self.hessian(w)

    def fuse_value_grad_hessian_vec_product(self, w, d):
        return self.value(w), self.grad(w), self.hessian_vec_product(w, d)


def make_oracle(data_path=None, sparse=False):
    if data_path is None:
        X, y, _ = generate_dataset(n=1000)
    else:
        X, y = load_svmlight_file(data_path)
        
        X = scipy.sparse.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
        y = LabelBinarizer().fit_transform(y)
    
    return Oracle(X, y)