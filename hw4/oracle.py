import scipy
import utils
import numpy as np

from scipy.special import expit
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import LabelBinarizer


class Oracle:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.L = 1.0
        
        self._call_count = 0

    @property
    def dim(self):
        return self.X.shape[1]
    
    def reset_oracle_count(self):
        self._call_count = 0

    def value(self, w):       
        self._call_count += 1 
        Xw = self.X @ w
        
        return -1/self.X.shape[0] * np.sum(self.y * np.log(expit(Xw) + 1e-12) + (1 - self.y) * np.log(1 - expit(Xw) + 1e-12))

    def grad(self, w):
        self._call_count += 1
        return (expit(self.X @ w) - self.y).T @ self.X / self.X.shape[0]

    def fuse_value_grad(self, w):
        self._call_count += 1
        
        N, sigmoid_Xw = self.X.shape[0], expit(self.X @ w)
        
        value = -1/N * np.sum(self.y * np.log(sigmoid_Xw + 1e-12) + (1 - self.y) * np.log(1 - sigmoid_Xw + 1e-12))
        grad = 1/N * (sigmoid_Xw - self.y).T @ self.X
        
        return value, grad


def make_oracle(data_path=None, sparse=False):
    if data_path is None:
        X, y, _ = utils.generate_dataset(n=1000, w_dim=100)
    else:
        X, y = load_svmlight_file(data_path)
        
        X = scipy.sparse.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
        y = LabelBinarizer().fit_transform(y)
    
    return Oracle(X, y)