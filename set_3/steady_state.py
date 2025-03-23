import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from matplotlib import cm

def discretize(L,N, source_location):
    M = np.zeros((N*N,N*N))
    b = np.zeros(N *N)
    r_c, t_c = source_location
    q = int(N // (2 / r_c))
    r_1 = np.linspace(0, r_c, q)
    r_2 = np.linspace(r_c, L, N-q+1)
    r = np.concatenate((r_1, r_2[1:]))
    p = np.linspace(t_c, 2*np.pi+t_c, N+1)
    R, P = np.meshgrid(r, p, indexing="ij")
    X, Y = R*np.cos(P), R*np.sin(P)
    dtheta = 2 * np.pi / N
    for i in range(2,N-1):
        for j in range(N):
            if not(i == q and j == 0):
                r_curr = r[i]
                dr_p = r_curr - r[i-1]
                dr_n = r[i+1] - r_curr
                idx = i * N + j
                next_r = (i+1)*N + j
                prev_r = (i-1)*N + j
                next_t = i * N + (j+1) % N
                prev_t = i * N + (j-1) % N

                M[idx, idx] = (-2 / (dr_p + dr_n)*(1/dr_p + 1/dr_n)) - 2 / (dtheta * dtheta)
                M[idx, next_r] = 2 / (dr_n * (dr_n + dr_p)) + 1/(r_curr * (dr_n + dr_p))
                M[idx, prev_r] = 2 / (dr_p * (dr_n + dr_p)) - 1/( r_curr * (dr_n + dr_p))
                M[idx, next_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
                M[idx, prev_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
            else:
                M[i * N + j, i * N + j] = 1
                b[q*N] = 1
    for j in range(N):
        i = 1
        r_curr = r[i]
        dr_p = r_curr - r[i-1]
        dr_n = r[i+1] - r_curr
        idx =  N + j
        next_r = 2 * N + j
        prev_r = 0 
        next_t = i * N + (j+1) % N
        prev_t = i * N + (j-1) % N

        M[idx, idx] = (-2 / (dr_p + dr_n)*(1/dr_p + 1/dr_n)) - 2 / (dtheta * dtheta)
        M[idx, next_r] = 2 / (dr_n * (dr_n + dr_p)) + 1/(r_curr * (dr_n + dr_p))
        M[idx, prev_r] = 2 / (dr_p * (dr_n + dr_p)) - 1/( r_curr * (dr_n + dr_p))
        print(2 / (dr_p * (dr_n + dr_p)) - 2/( r_curr * (dr_n + dr_p)))
        M[idx, next_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
        M[idx, prev_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
        M[0, 0] = -N / (dr_p * dr_p)
        M[0, j + N] = 1 / (dr_p * dr_p)
        idx_2 = (N-1) * N +j
        M[idx_2, idx_2] = 1
        if j != 0:
            M[j,j] = 1
    return M, b, X, Y


if __name__ == "__main__":
    R = 2
    N = 50
    source_coordinates = (np.sqrt(.6*.6 + 1.2*1.2), np.arctan(2))
    M, b, X, Y = discretize(R, N, source_coordinates)
    mask = np.all(M==0, axis=-1)
    Q = M[~mask]
    Q = Q[:,~mask]
    print(M)
    c = la.solve(M, b).reshape(N,N)
    z = np.zeros((N, N+1))
    z[:,:-1] = c
    z[:,-1] = c[:, 0]
    z[0,:] = np.ones(N+1) * c[0,0]
    fig = plt.figure()
    # ax = fig.add_subplot(projection="3d")
    # ax.plot_surface(X, Y, z, cmap = cm.coolwarm)
    plt.contourf(X, Y, z, vmin=0, vmax=1)
    plt.show()