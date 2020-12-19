import numpy as np
import matplotlib.pyplot as plt

def pos_def_matrix(N):
    A_ = np.random.randn(N, N)
    return A_ @ A_.T

def num_grad(f, x, eps=1e-8):
    grad = np.zeros(shape=(x.shape[0], 1))
    e = np.zeros(shape=(x.shape[0], 1))

    for i in range(x.shape[0]):
        e[i] = eps
        grad[i] = (f(x + e) - f(x - e)) / (2 * eps)
        e[i] = 0
    
    return grad

def num_hessian(df, x, eps=1e-8):
    hessian = np.zeros(shape=(x.shape[0], x.shape[0]))
    e = np.zeros(shape=(x.shape[0], 1))
    
    for i in range(x.shape[0]):
        e[i] = eps
        hessian[i, :] = (df(x + e) - df(x - e)).ravel() / (2 * eps)
        e[i] = 0
    
    return hessian


def plot_error(oracle, precision=np.float64):
    h = np.logspace(-18, 0, 100).astype(precision)
    errors = np.zeros_like(h).astype(precision)

    # np.random.seed(12)
    x = np.random.uniform(0.1, 1, size=(oracle.N, 1)).astype(precision)
    for i in range(errors.shape[0]):
        errors[i] = np.linalg.norm(num_grad(oracle.value, x, eps=h[i]) - oracle.grad(x), 1)

    eps = np.finfo(precision).eps

    plt.title(f"numerical differentiation error")
    plt.loglog(h, errors, label=f"{oracle.__class__.__name__}: {precision.__name__}")
    plt.axvline(eps, color="gray" if precision == np.float64 else "black", linestyle="--")
    plt.axvline(np.sqrt(eps), color="gray" if precision == np.float64 else "black", linestyle="--")
    plt.xlabel("$h$")
    plt.ylabel("abs error")
    # plt.show()


def check_task(oracle, x_type="vec", seed=42):
    np.random.seed(seed)
    
    x = np.random.uniform(size=(oracle.N, 1))
    
    print("--" * 8 + f" {oracle.__class__.__name__} "+ "--" * 8)
    print("gradient: ", np.linalg.norm(num_grad(oracle.value, x) - oracle.grad(x)))
    print("hessian: ", np.linalg.norm(num_hessian(oracle.grad, x) - oracle.hessian(x)))
    print()
    