import numpy as np 
from get_indices import get_indices
from scipy.linalg import eig
import matplotlib.pyplot as plt
from matplotlib import cm

L = 1
N = 4
M = np.zeros((N*N,N*N))
dr = L / N
dtheta = 2 * np.pi / N
for i in range(1,N-1):
    for j in range(N):
        idx = i * N + j
        next_r = (i+1)*N + j
        prev_r = (i-1)*N + j
        next_t = i * N + (j+1) % N
        prev_t = i * N + (j-1) % N

        r_curr = i * dr
        M[idx, idx] = -2 / (dr * dr) - 2 / (r_curr * r_curr * dtheta * dtheta)
        M[idx, next_r] = 1 / (dr * dr) + 1/(r_curr * dr)
        M[idx, prev_r] = 1 / (dr * dr) - 1/(r_curr * dr)
        M[idx, next_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
        M[idx, prev_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
print(M[4:12])