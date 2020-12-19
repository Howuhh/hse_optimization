import numpy as np
import matplotlib.pyplot as plt

from grad_check import pos_def_matrix
from grad_check import check_task, plot_error

class Task4_1:
    def __init__(self, N=5):
        self.A = pos_def_matrix(N)
        self.N = N
    
    def value(self, x):
        return 0.5 * np.linalg.norm(x @ x.T - self.A)**2
    
    def grad(self, x):
        return 2 * (x @ x.T @ x - self.A @ x)
    
    def hessian(self, x):    
        I = np.identity(self.N)
        return 2 * (x.T @ x) * I + 4 * (x @ x.T) - 2 * self.A

class Task4_2:
    def __init__(self, N=5):
        self.A = pos_def_matrix(N)
        self.N = N
    
    def value(self, x):
        return x.T @ (self.A @ x) / np.linalg.norm(x)**2
    
    def grad(self, x):
        x_norm = np.linalg.norm(x)
        return 2 * (self.A @ x) / x_norm**2 - 2 * (x @ x.T @ self.A @ x) / x_norm**4
    
    def hessian(self, x):
        x_norm = np.linalg.norm(x)
        A, I = self.A, np.identity(self.N)
                
        return 2 / x_norm**4 * (x_norm**2 * A - 2*A @ x @ x.T - x.T @ A @ x * I - 2*x @ x.T @ A + (4/x_norm**2) * x @ x.T @ A @ x @ x.T)
    
class Task4_3:
    def __init__(self, N=5):
        self.N = N

    def value(self, x):
        return (x.T @ x)**(x.T @ x)
    
    def grad(self, x):
        return 2 * (x.T @ x)**(x.T @ x) * (np.log(x.T @ x) + 1) * x
        
    def hessian(self, x):
        I = np.identity(self.N)
        
        return 2 * self.value(x) * (2 * ((np.log(x.T @ x) + 1) ** 2 + 1 / (x.T @ x)) * x @ x.T + (np.log(x.T @ x) + 1) * I)
        

class Task6_1:
    def __init__(self):
        self.N = 2
        
    def value(self, x):
        return 2*x[0]**2 + x[1]**2 * (x[0]**2 - 2)
    
    def grad(self, x):
        grad = np.zeros(shape=(2, 1))
        grad[0] = 4 * x[0] + 2 * x[1] ** 2 * x[0]
        grad[1] = 2 * x[0] ** 2 * x[1] - 4 * x[1]
        return grad
    
    def hessian(self, x):
        hessian = np.zeros(shape=(2, 2))
        hessian[0, 0] = 4 + 2 * x[1] ** 2
        hessian[0, 1] = hessian[1, 0] = 4 * x[0] * x[1]
        hessian[1, 1] = 2 * x[0] ** 2 - 4
        return hessian


class Task6_2:
    def __init__(self, lam=1.0):
        self.N = 2
        self.lam = lam
        
    def value(self, x):
        return (1 - x[0]) ** 2 + self.lam * (x[1] - x[0] ** 2) ** 2

    def grad(self, x):
        grad = np.zeros(shape=(self.N, 1))
        grad[0] = 2 * (x[0] - 1 - 2 * self.lam * x[0] * x[1] + 2 * self.lam * x[0] ** 3)
        grad[1] = 2 * self.lam * (x[1] - x[0] ** 2)
        return grad

    def hessian(self, x):
        hessian = np.zeros(shape=(self.N, self.N))
        hessian[0, 0] = 2 * (1 + 6 * self.lam * x[0] ** 2 - 2 * self.lam * x[1])
        hessian[0, 1] = hessian[1, 0] = -4 * self.lam * x[0]
        hessian[1, 1] = 2 * self.lam
        return hessian


class Task7_1:
    def __init__(self, N=5):
        self.N = N
    
    def value(self, x):
        return -np.sum(x * np.log(np.maximum(x, 1e-16)))
        
    def grad(self, x):
        return -np.log(np.maximum(x, 1e-16)) - 1
        

class Task7_2:
    def __init__(self):
        self.N = 1
    
    def value(self, x):
        return x**3
        
    def grad(self, x):
        return 3*x**2
        

def main():
    N = 5
    
    check_task(Task4_1(N))
    check_task(Task4_2(N))
    check_task(Task4_3(N))
    
    check_task(Task6_1())
    check_task(Task6_2())
    
    plt.figure(figsize=(12, 8))
    plot_error(Task7_1(), precision=np.float32)
    plot_error(Task7_1(), precision=np.float64)
    
    plot_error(Task7_2(), precision=np.float32)
    plot_error(Task7_2(), precision=np.float64)
    
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()