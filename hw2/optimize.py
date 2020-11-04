import numpy as np
import step_search


def optimize(oracle, start_point, line_search_method="golden", tol=1e-8, max_iter=10000, verbose=True):
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
        
        if verbose and i % 1000 == 0:
            print(f"Iteration {i}: {fi}, alpha: {alpha}, grads: {np.linalg.norm(grad)**2 / grad0_norm}")
        
        if np.linalg.norm(grad)**2 / grad0_norm <= tol:
            break

    return w
        
        
def main():
    pass
    
    
if __name__ == "__main__":
    main()
    
