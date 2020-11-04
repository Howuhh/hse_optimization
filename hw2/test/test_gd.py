import sys
import numpy as np

sys.path.append('..')

from oracle import make_oracle
from cross_entropy import sigmoid
from optimize import optimize

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def _test(data_path=None, line_search_method="golden", tol=1e-8):
    oracle = make_oracle(data_path=data_path)

    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    
    w_pred = optimize(oracle, w_init, line_search_method=line_search_method, tol=tol, verbose=False)
    y_pred = np.round(sigmoid(oracle.X @ w_pred))
    
    print(f"Test {line_search_method}, data: {data_path if data_path else 'generated'}")
    print("--"*35)
    print("Cross entropy: ", oracle.value(w_pred))
    print("ROC AUC score: ", roc_auc_score(oracle.y, y_pred))
    
    model = LogisticRegression(penalty="none", tol=1e-8, max_iter=10000, n_jobs=-1, fit_intercept=False)
    model.fit(oracle.X, oracle.y.ravel())

    print("Cross entropy (sklearn): ", oracle.value(model.coef_.reshape(-1, 1)))
    print("ROC AUC score (sklearn): ", roc_auc_score(oracle.y, model.predict(oracle.X)))
    
    print(f"Difference: {abs(oracle.value(w_pred) - oracle.value(model.coef_.reshape(-1, 1)))}")
    print()
    

def main():   
    _test(line_search_method="golden") 
    _test(line_search_method="brent")
    
    _test("../data/a1a.txt", line_search_method="golden") 
    _test("../data/a1a.txt", line_search_method="brent")
    
    _test("../data/breast-cancer_scale.txt", line_search_method="golden") 
    _test("../data/breast-cancer_scale.txt", line_search_method="brent")


if __name__ == "__main__":
    main()