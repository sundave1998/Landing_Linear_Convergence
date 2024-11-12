import numpy as np
from update_utils import *
from tqdm import tqdm
import copy
import time

def RDGD_tracking(A, W, step_size, X_0, x_opt, max_iter=10000, lin_term=None, verbose=False):
    distance = []
    con_errors = []
    grad_norms = []
    N = X_0.shape[0]
    n = X_0.shape[1]
    r = X_0.shape[2]
    
    beta = 1
    x = np.copy(X_0)
    y = np.zeros_like(X_0)
    grad = np.zeros_like(X_0)
    # old_grad = np.zeros_like(X_0)
    diff = np.zeros_like(X_0)
    # penalty = np.zeros_like(X_0)
    
    A_m = np.zeros((n,n))
    for i in range(N):
        A_m += A[i]@A[i].T
    
    all_start = time.time()
    retraction_time = 0
    communication_time = 0
    gradient_time = 0
    logging_time = 0

    for i in range(N):
        diff[i] = A[i]@A[i].T@x[i]
        if lin_term is not None:
            diff[i] += lin_term[i]
        grad[i] = proj_tangent(x[i], diff[i])
        # penalty[i] = lambd * x[i] @ (x[i].T@x[i] - np.eye(r))
        # y[i] = proj_tangent(x[i], diff[i])
        y[i] = grad[i] 

    for k in tqdm(range(max_iter)):
        old_grad = copy.deepcopy(grad)
        # old_diff = copy.deepcopy(diff)
        old_x = copy.deepcopy(x)
        old_y = copy.deepcopy(y)
        # old_penalty = copy.deepcopy(penalty)
        start = time.time()
        Wx = np.zeros_like(x)
        for i in range(N):
            for j in range(N):
                Wx[i] += W[i,j] * (old_x[j] - old_x[i])
                # Wx[i] += W[i,j] * (old_x[j])
        end = time.time()
        communication_time += end-start
        
        
        
        start = time.time()
        for i in range(N):
            dx = proj_tangent(x[i], beta*Wx[i] + step_size*y[i])
            new_x = x[i] + dx
            U, S, Vh = np.linalg.svd(new_x, full_matrices=True)
            x[i] = U[:,:r]@Vh
        end = time.time()
        retraction_time += end-start
        
        
        
        start = time.time()
        grad = np.zeros_like(X_0)
        diff = np.zeros_like(X_0)
        for i in range(N):
            diff[i] = A[i]@A[i].T@x[i]
            if lin_term is not None:
                diff[i] += lin_term[i]          
            grad[i] = proj_tangent(x[i], diff[i])
        end = time.time()
        gradient_time += end-start
        
        
        start = time.time()
        y = np.zeros_like(old_y)
        for i in range(N):
            y[i] = old_y[i]
            for j in range(N):
                y[i] += beta*W[i,j] * (old_y[j] - old_y[i])
            y[i] += (grad[i] - old_grad[i])
        end = time.time()
        communication_time += end-start
        
        start = time.time()
        x_bar = np.average(x, axis=0)
        # u, s, v = np.linalg.svd(x_opt.T@x_bar)
        # dist = np.sqrt(2*np.abs(r - np.sum(s)))
        dist = 0
        for i in range(N):
            dist += np.linalg.norm(x[i].T@x[i]- np.eye(r), ord='fro')  
            
                                  
        grad_avg = np.average(grad, axis=0)
        grad_norm = np.linalg.norm(grad_avg)
        
        x_bar = np.average(x, axis=0)
        consensus_error = 0
        for i in range(N):
            consensus_error += np.linalg.norm(x[i]- x_bar)
            
        if (consensus_error+grad_norm+dist) < (1e-10*grad_avg.shape[0]*grad_avg.shape[1]) or np.isnan(dist):
            break

        if verbose and k%100 == 0:
            x_opt, s, v = np.linalg.svd(A_m)
            x_opt = x_opt[:, :r]
            print("iter:", k, "dist:", dist)
            print("average x:", np.linalg.norm(np.average(x, axis=0)-x_opt))
    
            # return np.zeros(max_iter)
            
            print("consensus error:", consensus_error)
            print("function value:")
            print(np.sum(np.sum(x_bar.T@A_m@x_bar)))
            print("dist", dist)
        distance.append(dist)
        grad_norms.append(grad_norm)
        con_errors.append(consensus_error)
        
        end = time.time()
        logging_time += end-start
    # print(x_bar.T@x_bar)
    # print("average x:")
    # print(x_bar)
    total_time = time.time()-all_start
    print("total time:", total_time)
    print("retraction_time time:", retraction_time)
    print("communication_time time:", communication_time)
    print("gradient time:", gradient_time)
    print("logging_time time:", logging_time)
    return np.array(distance), np.array(con_errors), np.array(grad_norms), x_bar

