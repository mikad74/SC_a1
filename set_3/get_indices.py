import numpy as np
def get_indices(N):
    indices = list()
    for i in range(N*N):
        if i%N ==0:
            pass
        elif (i+1+N) % N == 0 :
            pass
        elif (i < N or i >= (N*(N-1))):
            pass
        else:
            indices.append(i)
    return np.array(indices)


if __name__ == "__main__":
    for N in range(4, 400, 2):
        T = np.arange(N*N).reshape(N,N)
        indices = get_indices(N)
        assert (indices.reshape(N-2, N-2) - T[1:-1, 1:-1] == np.zeros((N-2,N-2))).all()
