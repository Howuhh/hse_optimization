import numpy as np
import step_search

from oracle import make_oracle
from cross_entropy import sigmoid

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def optimize(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000):
    line_search = getattr(step_search, f"{line_search_method}_line_search")
    
    grad0 = oracle.grad(start_point)
    grad0_norm = np.linalg.norm(grad0)**2
    
    w = start_point
    for i in range(1, max_iter):
        fi, grad = oracle.fuse_value_grad(w)
        
        grad = grad.reshape(-1, 1)

        assert(w.shape == grad.shape), "diff shape for grad and w"
        
        alpha = line_search(oracle, w, grad)
        w = w - alpha * grad.reshape(-1, 1)
        
        if i % 10 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2 / grad0_norm}")
        
        if np.linalg.norm(grad)**2 / grad0_norm <= tol:
            break

    return w
        
        
def main():
    oracle = make_oracle("data/a1a.txt")
    
    # https://leimao.github.io/blog/Weights-Initialization/
    w_n = oracle.X.shape[1]
    w_init = np.random.uniform(-1/np.sqrt(w_n), 1/np.sqrt(w_n), size=w_n).reshape(-1, 1)
    
    w_pred = optimize(oracle, w_init, line_search_method="brent")
    y_pred = np.round(sigmoid(oracle.X @ w_pred))
    
    print(confusion_matrix(oracle.y, y_pred.flatten()))
    print("Cross entropy: ", oracle.value(w_pred))
    print("ROC AUC score: ", roc_auc_score(oracle.y, y_pred))
    
    model = LogisticRegression(penalty="none", tol=1e-8, max_iter=10000, n_jobs=-1, fit_intercept=False)
    model.fit(oracle.X, oracle.y.ravel())
    
    print("Cross entropy (sklearn): ", oracle.value(model.coef_.reshape(-1, 1)))
    print("ROC AUC score (sklearn): ", roc_auc_score(oracle.y, model.predict(oracle.X)))
    
    
if __name__ == "__main__":
    main()
    
