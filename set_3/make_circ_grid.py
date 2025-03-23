import numpy as np
import matplotlib.pyplot as plt


def make_grid(N, L, coordinates):
    r_c, t_c = coordinates
    q = int(N // (2 / r_c))
    print(q, N - q)
    r_1 = np.linspace(0, r_c, q)
    r_2 = np.linspace(r_c, L, N-q+1)
    r = np.concatenate((r_1, r_2[1:]))
    # r = np.linspace(0, L, N)
    p = np.linspace(t_c, 2*np.pi+t_c, N+1)
    R, P = np.meshgrid(r, p, indexing="ij")
    X, Y = R*np.cos(P), R*np.sin(P)
    return X, Y

if __name__ == "__main__":
    L = 2
    for N in range(5, 10, 1):
        X, Y = make_grid(N, L, (np.sqrt(.6*.6 + 1.2 * 1.2), np.arctan(2)))
        plt.scatter(X,Y)
        plt.scatter(.6, 1.2, color="k")
        plt.show()
