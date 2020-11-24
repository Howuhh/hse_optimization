import sys
import numpy as np

sys.path.append('..')
np.random.seed(42)

from oracle import make_oracle
from tabulate import tabulate
from optimize import optimize_gd, optimize_hfn, optimize_newton


def print_report(config, data_path=None):
    print()
    oracle = make_oracle(data_path=data_path)
    w_init = np.zeros((oracle.X.shape[1], 1))
    
    for method in config:
        optimizer = config[method]["optimizer"]
        line_search_methods = config[method]["line_search_methods"]
        
        table = [["line search", "entropy", "num iter", "oracle calls", "time, s"]]
        
        for line_search in line_search_methods:
            w, log = optimizer(oracle, w_init, line_search, tol=1e-8, max_iter=10000, verbose=False)
            info = log.get_log()
    
            table.append([line_search, info["entropy"][-1], info["num_iter"], info["oracle_calls"][-1], round(info["time"][-1], 2)])
    
        print(f"- {method.upper()} " + "-" * (68 - len(method)))
        print(tabulate(table, tablefmt="github", headers="firstrow"))
        print(f"-" * 71)
        print()


def main():
    config = {
        "gradient descent": {
            "optimizer": optimize_gd,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe", "lipschitz"]
        },
        "newton": {
            "optimizer": optimize_newton,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe"]
        },
        "hf_newton": {
            "optimizer": optimize_hfn,
            "line_search_methods": ["golden", "brent", "armijo", "wolfe"]
        }
    }
    
    print_report(config, "../data/a1a.txt")
    # print_report(config)
    # print_report(config, "../data/breast-cancer_scale.txt")
    
    
if __name__ == "__main__":
    main()
