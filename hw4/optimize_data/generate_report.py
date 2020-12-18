import sys
import numpy as np

sys.path.append('..')
np.random.seed(42)

from oracle import make_oracle
from tabulate import tabulate
from optimize_lasso import optimize_lasso


def file_name(path):
    if path is None:
        return "synthetic"
    return path.split("/")[-1].split(".")[0] 


def print_report(config, data_path=None):
    oracle = make_oracle(data_path=data_path)
    w_init = np.zeros((oracle.dim, 1))
    
    table = [["method", "entropy", "num iter", "oracle calls", "time, s"]]
    
    optimizer = config["optimizer"]
    
    for lambda_ in config["lambda_"]:
        w, log = optimizer(oracle, w_init, tol=1e-8, max_iter=10000, lambda_=lambda_)
        
        method_name = f"{config['method']} (lambda={lambda_})"
        table.append([method_name, *log.get_log_last()[:-1]])
    
    table = tabulate(table, tablefmt="github", headers="firstrow")
    
    print(file_name(data_path).upper())
    print("-" * table.find("\n"))
    print(table)
    print("-" * table.find("\n"))
    print()


def all_methods():
    config = {
        "method": "ProximalGD",
        "optimizer": optimize_lasso,
        "lambda_": [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0]
    }
    
    print_report(config, "../data/a1a.txt")
    print_report(config, "../data/breast-cancer_scale.txt")
    print_report(config)
    
    
if __name__ == "__main__":
    all_methods()