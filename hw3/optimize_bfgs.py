import numpy as np
from time import time

from collections import deque
from utils import run_optimizer
from step_search import wolfe_line_search
from optimize_data.optimize_log import OptimizeLog


def bfgs_update(H, I, s, y):
    p = 1 / (y.T @ s)
    V = I - p * np.outer(y, s)
    
    H_new = V.T @ H @ V + p * np.outer(s, s)
    
    return H_new


def optimize_bfgs(oracle, w, tol=1e-8, gamma=1.0, max_iter=10000, verbose=False):
    log = OptimizeLog(start_time=time())
    oracle.reset_oracle_count()
    
    I = np.identity(oracle.dim)
    H = gamma * I
    
    prev_w, prev_grad = w, oracle.grad(w).reshape(-1, 1)
    grad0_norm = prev_grad.T @ prev_grad
    for iter_ in range(max_iter):
        grad_norm = (prev_grad.T @ prev_grad) / grad0_norm
        if grad_norm <= tol:
            break
        
        direction = H @ prev_grad
        alpha = wolfe_line_search(oracle, w, direction, not_converge="armijo")
        
        # works only for near to zeros start points (from 100+ iterations to 80)
        # but, may not converge sometimes (so, not a huge win)
        # if iter_ == 0: 
            # alpha = 1.0

        w = w - alpha * direction
        
        new_grad = oracle.grad(w).reshape(-1, 1)        
        s = w - prev_w
        y = new_grad - prev_grad
        
        if iter_ == 0:
            H = ((y.T @ s) / (y.T @ y)) * H
        
        H = bfgs_update(H, I, s, y)        
        prev_w, prev_grad = w, new_grad
        
        entropy = oracle.value(w)
        if verbose and iter_ % 1 == 0:
            print(f"Iteration {iter_}: {entropy}, alpha: {alpha}, grads: {grad_norm}")
        
        log.add_log(time(), entropy, alpha, grad_norm, oracle._call_count)
        
    return w, log


def lbfgs_direction(grad, gamma, s_q, y_q, r_q):
    alpha = np.zeros(len(s_q))

    q = grad
    for i in range(len(s_q) - 1, -1, -1):
        alpha[i] = r_q[i] * s_q[i].T @ q
        q = q - alpha[i] * y_q[i]
        
    Hd = gamma * q
    for i in range(len(s_q)):
        beta = r_q[i] * y_q[i].T @ Hd
        Hd = Hd + s_q[i] * (alpha[i] - beta)    
    
    return Hd


def optimize_lbfgs(oracle, w, tol=1e-8, buffer_size=5, gamma=1.0, max_iter=10000, verbose=False):
    log = OptimizeLog(start_time=time())
    oracle.reset_oracle_count()
    
    s_q, y_q, r_q = [deque(maxlen=buffer_size) for _ in range(3)]
    
    prev_w, prev_grad = w, oracle.grad(w).reshape(-1, 1)
    grad0_norm = prev_grad.T @ prev_grad
    
    for iter_ in range(max_iter):
        grad_norm = (prev_grad.T @ prev_grad) / grad0_norm
        if grad_norm <= tol:
            break
        
        if iter_ != 0:
            gamma = (y_q[-1].T @ s_q[-1]) / ((y_q[-1].T @ y_q[-1]) + 1e-14)
            
        direction = lbfgs_direction(prev_grad, gamma, s_q, y_q, r_q)
        alpha = wolfe_line_search(oracle, w, direction, not_converge="armijo")

        w = w - alpha * direction
        new_grad = oracle.grad(w).reshape(-1, 1)
        
        s_q.append(w - prev_w)
        y_q.append(new_grad - prev_grad)
        r_q.append(1 / ((y_q[-1].T @ s_q[-1]) + 1e-14)) # div by zero ???
        
        prev_w, prev_grad = w, new_grad
        
        entropy = oracle.value(w)
        if verbose and iter_ % 1 == 0:
            print(f"Iteration {iter_}: {entropy}, alpha: {alpha}, grads: {grad_norm}")
        
        log.add_log(time(), entropy, alpha, grad_norm, oracle._call_count)
        
    return w, log


def main():
    # w, log = run_optimizer(None, optimize_bfgs, gamma=10, verbose=True)
    w, log = run_optimizer(None, optimize_lbfgs, gamma=1.0, buffer_size=100, verbose=True)
    

if __name__ == "__main__":
    main()