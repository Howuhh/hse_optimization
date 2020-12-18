import oracle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix


def generate_dataset(n=50, w_dim=1, sparse=False):
    X = np.random.normal(size=(n, w_dim))
    X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
    
    w = np.random.uniform(-1, 1, size=(w_dim + 1, 1)).astype(np.float64)
    
    y = (X @ w >= 0).astype(np.int)
    
    if sparse:
        X = csr_matrix(X)

    return X, y, w


def run_optimizer(data_path, optimizer, **kwargs):
    oracle_ = oracle.make_oracle(data_path)
    w_init = np.zeros(oracle_.dim).reshape(-1, 1)
    
    w, log = optimizer(oracle_, w_init, **kwargs)
    return w, log


def plot_weights(oracle, optimize, penalty_range, names=None, intercept=True):
    weight_path = np.zeros((oracle.dim, len(penalty_range)))

    w_init = np.zeros((oracle.dim, 1))
    for i, penalty in enumerate(penalty_range):
        w = optimize(oracle, w_init, lambda_=penalty)
        
        weight_path[:, i] = w.ravel()

    if not intercept:
        weight_path = weight_path[:-1]
        
    plt.figure(figsize=(12, 8))
    for i, w in enumerate(weight_path):
        plt.plot(penalty_range, w)
    
    plt.xlabel("$\lambda$")
    plt.ylabel("$W_{i}$")
    plt.grid()


# TODO: rewrite
def plot_metric(log, data, xaxis, offset=0):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    for method in log[data]:
        if method == "Gradient Descent":
            continue
        
        method_log = log[data][method]
        info = method_log.get_log()
        
        error = np.log10(method_log.error)[offset:]
        grads = np.log10(info["grad_info"])[offset:]
        
        if xaxis == "num_iter":
            metric = np.arange(error.shape[0])
        else:
            metric = info[xaxis][offset:]
        
        ax[0].plot(metric, error, label=f"{method}: {round(info['entropy'][-1],6)}")
        ax[1].plot(metric, grads.ravel())
        
    ax[0].set(title="Convergence by $|F(w^*) - F(w)|$", 
              xlabel=xaxis, ylabel="$\\log_{10} |F(w^*) - F(w)|$")
    ax[1].set(title="Convergence by $\\frac{|\\nabla F(w)|}{|\\nabla F(w_0)|}$",
              xlabel=xaxis, ylabel="$\\log_{10} \\frac{|\\nabla F(w)|}{|\\nabla F(w_0)|}$")
    
    ax[0].legend()


if __name__ == "__main__":
    print(generate_dataset(5))