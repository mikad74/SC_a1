import numpy as np 
from get_indices import get_indices
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm

def get_eigenmodes(L, N, shape="square"):
    M = np.zeros((N*N,N*N))
    if shape == "square" or shape == "rect":
        indices = get_indices(N)
        if shape == "square":
            dy = L/N
            dx = L/N
            X, Y = np.meshgrid(np.arange(N)/N*L, np.arange(N)/N*L)
        else:
            dy = 2*L/N
            dx = L/N
            X, Y = np.meshgrid(np.arange(N)/N*L, np.arange(N)/N*2*L)
        for i in indices:
            q = M[i]
            q[i-N] = 1/(dy*dy)
            q[i+N] = 1/(dy*dy)
            q[i-1] = 1/(dx*dx)
            q[i+1] = 1/(dx*dx)
            q[i] = -2*(1/(dx*dx) + 1/(dy*dy))
    else:
        dr = L / (N*2)
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
        for j in range(N):
            next_r = N + j
            M[j, j] = 1
            M[j, next_r] = -1
        r = np.linspace(0, L, N)
        p = np.linspace(0, 2 * np.pi, N)
        R, P = np.meshgrid(r, p)
        X, Y = R *np.cos(P), R*np.sin(P)
    eigval, eigvec = eig(M)
    eigval = np.sqrt(eigval*-1)
    return eigval, eigvec, (X, Y)

def plot_eigenmodes(eigval, eigvec, X, Y, N, L):
    n_plots = 4
    fig = plt.figure()
    smallest_evs = np.argsort(np.where(eigval.real != 0, eigval, np.inf))[:n_plots]
    print(smallest_evs)
    for q, i in enumerate(smallest_evs):
        o = 221 + q
        print(o)
        ax = fig.add_subplot(o, projection="3d")
        ax.plot_surface(X, Y, eigvec[:, i].reshape(N,N), label=f"{eigval[i]}", cmap=cm.coolwarm, linewidth=0, antialiased = False)
        # print(eigvec[:, i].reshape(N,N))
        plt.legend()
    p = eigval.real
    p_ = np.where(p != 0, p, np.inf)
    # print(np.argsort(p_))
    # print(eigval[np.argsort(p_)[:5]])
    plt.show()

def animate_eigenmodes(eigval, eigvec, X, Y, N, L, rank=0):
    fig = plt.figure()
    smallest_evs = np.argsort(np.where(eigval.real != 0, eigval, np.inf))
    i = smallest_evs[rank]
    ax = fig.add_subplot(111, projection='3d')
    artists = []
    timeline = np.linspace(0, 100)
    for t in timeline:
        container = [ax.plot_surface(X, Y, eigvec[:,i].reshape(N,N) * np.cos(eigval[i] * t), cmap = cm.coolwarm, linewidth = 0 , antialiased=False)]
        artists.append(container)
    ani = animation.ArtistAnimation(fig = fig, artists=artists)
    ani.save("membrane.gif")
    plt.show()

    pass

if __name__ == "__main__":
    L = 100
    N = 30
    eigval, eigvec, (X, Y) = get_eigenmodes(L, N, shape="rect")
    animate_eigenmodes(eigval, eigvec, X, Y, N, L, 1)
    # plot_eigenmodes(eigval, eigvec, X, Y, N, L)