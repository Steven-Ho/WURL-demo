# Wasserstein distance based pseudo-reward estimation
import numpy as np

def match(A, B, D ,T):
    count_A = [T for _ in range(D)]
    count_B = [D for _ in range(T)]
    distance = np.zeros(D)
    i = j = 0
    while i<D:
        if count_A[i] == 0:
            i += 1
        elif count_B[j] == 0:
            j += 1
        else:
            delta = min(count_A[i], count_B[j])
            count_A[i] -= delta
            count_B[j] -= delta
            distance[i] += np.linalg.norm(A[i] - B[j], ord=2, axis=-1)*delta/T
    return distance

def wsre(A, B):
    D, d = A.shape
    T = B.shape[0]
    M = 5
    # d: data dimension, D: number of source data points, T: number of benchmark data points, M: number of v
    mean = np.zeros(d)
    cov = np.eye(d)
    wd = np.zeros((M, D))
    v = np.random.multivariate_normal(mean, cov, M)
    l = 1./np.linalg.norm(v, ord=2, axis=-1)
    v = v * l[:, None]
    for i in range(M):
        pA = np.matmul(A, v[i])
        pB = np.matmul(B, v[i])
        iA = np.argsort(pA)
        iB = np.argsort(pB)
        A = A[iA]
        B = B[iB]
        m = match(A, B, D, T)
        wd[i, iA] = m
    return np.mean(wd, axis=0)