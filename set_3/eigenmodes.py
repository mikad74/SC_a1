import numpy as np 
from get_indices import get_indices
from scipy.linalg import eig
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import time

def get_eigenmodes(L, N, shape="square", sparse=False):
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
        for i in range(2,N-1):
            for j in range(N):
                idx = i * N + j
                next_r = (i+1)*N + j
                prev_r = (i-1)*N + j
                next_t = i * N + (j+1) % N
                prev_t = i * N + (j-1) % N

                r_curr = i * dr
                M[idx, idx] = -2 / (dr * dr) - 2 / (r_curr * r_curr * dtheta * dtheta)
                M[idx, next_r] = 1 / (dr * dr) + 1/(2 * r_curr * dr)
                M[idx, prev_r] = 1 / (dr * dr) - 1/(2 * r_curr * dr)
                M[idx, next_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
                M[idx, prev_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
        for j in range(N):
            idx = N + j
            next_r = (2)*N + j
            prev_r = 0
            next_t = i * N + (j+1) % N
            prev_t = i * N + (j-1) % N

            r_curr = i * dr
            M[idx, idx] = -2 / (dr * dr) - 2 / (r_curr * r_curr * dtheta * dtheta)
            M[idx, next_r] = 1 / (dr * dr) + 1/(r_curr * dr)
            M[idx, prev_r] = 1 / (dr * dr) - 1/(r_curr * dr)
            M[idx, next_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
            M[idx, prev_t] = 1 / (r_curr * r_curr * dtheta * dtheta)
            M[0, 0] = -N / (dr * dr)
            M[0, j + N] = 1 / (dr * dr)
        r = np.linspace(0, L/2, N)
        p = np.linspace(0, 2 * np.pi, N+1, endpoint=True)
        R, P = np.meshgrid(r, p, indexing="ij")
        X, Y = R *np.cos(P), R*np.sin(P)
        print(M[0])
    if sparse:
        eigval, eigvec = eigsh(M)
    else:
        eigval, eigvec = eig(M)
    eigval = np.sqrt(-1*eigval)
    return eigval, eigvec, (X, Y)

def plot_eigenmodes(eigval, eigvec, X, Y, N, L):
    n_plots = 4
    fig = plt.figure()
    smallest_evs = np.argsort(np.where(eigval.real != 0, eigval, np.inf))[:n_plots]
    for q, i in enumerate(smallest_evs):
        o = 221 + q
        ax = fig.add_subplot(o, projection="3d")
        if (N,N) != X.shape:
            extended = np.zeros((N, N +1))
            extended[:,:-1] = eigvec[:,i].reshape(N,N)
            extended[:,-1] = eigvec[:,i].reshape(N,N)[:,0]
            origin = np.ones_like(extended[0]) * extended[0,0]
            extended[0] = origin
            print(eigvec[:,i].reshape(N,N))
            ax.plot_surface(X, Y, extended, label=f"{np.round(eigval[i].real,2)} [Hz]", cmap=cm.coolwarm, linewidth=0, antialiased = False)
        else:
            ax.plot_surface(X, Y, eigvec[:, i].reshape(N,N), label=f"{np.round(eigval[i].real,2)} [Hz]", cmap=cm.coolwarm, linewidth=0, antialiased = False)
        plt.legend()
    p = eigval.real
    plt.show()

def animate_eigenmodes(eigval, eigvec, X, Y, N, L, rank=0, name="membrane"):
    fig = plt.figure()
    smallest_evs = np.argsort(np.where(eigval.real != 0, eigval, np.inf))
    i = smallest_evs[rank]
    ax = fig.add_subplot(111, projection='3d')
    artists = []
    timeline = np.linspace(0, 5, 500)
    if (N,N) != X.shape:
        reshaped = np.zeros((N, N +1))
        reshaped[:,:-1] = eigvec[:,i].reshape(N,N)
        reshaped[:,-1] = eigvec[:,i].reshape(N,N)[:,0]
        origin = np.ones_like(reshaped[0]) * reshaped[0,0]
        reshaped[0] = origin
    else:
        reshaped = eigvec[:,i].reshape(N,N)
        print(reshaped[-5])
    for t in timeline:
        container = [ax.plot_surface(X, Y, reshaped * np.cos(eigval[i] * t), cmap = cm.coolwarm, linewidth = 0 , antialiased=False)]
        artists.append(container)
    ani = animation.ArtistAnimation(fig = fig, artists=artists, interval=10)
    ani.save(f"{name}.gif")
    plt.show()

    pass

if __name__ == "__main__":
    L = 1
    N = 50
    # Plot square eigenmodes
    eigval, eigvec, (X, Y) = get_eigenmodes(L, N, shape="square")
    plot_eigenmodes(eigval, eigvec, X, Y, N, L)
    animate_eigenmodes(eigval, eigvec, X, Y, N, L, 0, "eigenmodes")
    # Plot rectangular eigenmodes
    eigval, eigvec, (X, Y) = get_eigenmodes(L, N, shape="rect")
    plot_eigenmodes(eigval, eigvec, X, Y, N, L)
    # Plot circle eigenmodes
    eigval, eigvec, (X, Y) = get_eigenmodes(L, N, shape="circle")
    plot_eigenmodes(eigval, eigvec, X, Y, N, L)
    L_values = [1, 10, 100]
    plt.cla()
    for L in L_values:
        time_s_start = time.time()
        eigval, _, __ = get_eigenmodes(L, N, shape='square', sparse=True)
        time_s_stop = time.time()
        print(f"sparse method with size L = {L}, time: {time_s_stop - time_s_start}")
        time_n_start = time.time()
        eigval, _, __ = get_eigenmodes(L, N, shape='square', sparse=False)
        time_n_stop = time.time()
        print(f"non-sparse method with size L = {L}, time: {time_n_stop - time_n_start}")
        plt.yscale("log")
        plt.xlabel("Rank")
        plt.ylabel("Frequency [Hz]")
        plt.plot(np.sort(eigval)[::-1], label=f"L = {L}")
    plt.legend()
    plt.show()


    # animate_eigenmodes(eigval, eigvec, X, Y, N, L, 1)