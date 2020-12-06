import sys
import numpy as np


sys.path.append('..')
np.random.seed(42)

from oracle import make_oracle
from tabulate import tabulate
from optimize import optimize_gd, optimize_hfn, optimize_newton
from optimize_bfgs import optimize_bfgs, optimize_lbfgs


def file_name(path):
    if path is None:
        return "synthetic"
    return path.split("/")[-1].split(".")[0] 


def print_report(config, data_path=None):
    # print()
    oracle = make_oracle(data_path=data_path)
    w_init = np.zeros((oracle.dim, 1))
    
    table = [["method", "entropy", "num iter", "oracle calls", "time, s"]]
    
    for method in config:
        optimizer, params = config[method]["optimizer"], config[method].get("params", {})
        
        w, log = optimizer(oracle, w_init, tol=1e-8, max_iter=10000, **params)
        info = log.get_log()
    
        table.append([method, round(info["entropy"][-1], 16), info["num_iter"], info["oracle_calls"][-1], round(info["time"][-1], 2)])
    
    table = tabulate(table, tablefmt="github", headers="firstrow")
    
    print(file_name(data_path).upper())
    print("-" * table.find("\n"))
    print(table)
    print("-" * table.find("\n"))
    print()


def main():
    config = {
        "Gradient Descent (armijo)": {
            "optimizer": optimize_gd,
            "params": {
                "line_search_method": "armijo"
            }
        },
        "Newton (wolfe)": {
            "optimizer": optimize_newton,
            "params": {
                "line_search_method": "wolfe"
            }
        },
        "Hessian-Free Newton (armijo)": {
            "optimizer": optimize_hfn,
            "params": {
                "line_search_method": "armijo"
            }
        },
        "BFGS (wolfe)": {
            "optimizer": optimize_bfgs,
            "params": {
                "gamma": 20.0
            }
        },
        "L-BFGS (wolfe)": {
            "optimizer": optimize_lbfgs,
            "params": {
                "buffer_size": 100,
                "gamma": 1.0
            }
        }
    }
    
    print_report(config, "../data/a1a.txt")
    print_report(config, "../data/breast-cancer_scale.txt")
    print_report(config)
    
    
if __name__ == "__main__":
    main()
