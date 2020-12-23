import oracle
import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import csr_matrix


def file_name(path):
    if path is None:
        return "synthetic"
    return path.split("/")[-1].split(".")[0] 


def generate_dataset(n=50, w_dim=1, sparse=False):
    X = np.random.normal(size=(n, w_dim))
    X = np.hstack((X, np.ones(X.shape[0]).reshape(-1, 1)))
    
    w = np.random.uniform(-1, 1, size=(w_dim + 1, 1)).astype(np.float64)
    
    y = (X @ w >= 0).astype(np.int)
    
    if sparse:
        X = csr_matrix(X)

    return X, y, w


def plot_convergence(log):
    fig, ax = plt.subplots(1, 2, figsize=(18, 8))
    
    iters = np.arange(len(log["best_run_entropy"]))
    
    ax[0].plot(iters, np.log10(log["best_run_entropy"]), label=f"best entropy: {round(log['best_run_entropy'][-1],6)}")
    ax[1].plot(iters, np.log10(log["best_run_grad"]))

    ax[0].set(title="Convergence by $F(w)$", xlabel="iteration", ylabel="$F(w)$")
    ax[1].set(title="Convergence by stopping criterion", xlabel="iteration", ylabel="stopping criterion")

    ax[0].legend()


def plot_lr(log):
    fig, ax = plt.subplots(3, 1, figsize=(12, 14))
    
    ylabels = ["Time, s", "Number of iterations", "Number of nonzero weights"]
    
    for i, metric in enumerate(["time", "iters", "nonzero"]):
        ax[i].plot(log["lr"], log[metric])
        ax[i].set(xlabel="$\lambda$", ylabel=ylabels[i])
        ax[i].set_xscale("log")


def plot_weights(log, intercept=False):
    weight_path = log["weights_path"]
    
    if not intercept:
        weight_path = weight_path[:-1]
        
    plt.figure(figsize=(12, 8))
    for i, w in enumerate(weight_path):
        plt.plot(log["lr"], w)
    
    plt.xscale("log")
    plt.xlabel("$\lambda$")
    plt.ylabel("$W_{i}$")


if __name__ == "__main__":
    print(generate_dataset(5))