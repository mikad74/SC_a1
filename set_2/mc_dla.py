import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm


def add_walker(grid, p=1):
    i = np.random.choice(np.arange(grid.shape[0]))
    j = 0
    directions = np.array([(1,0), (-1,0), (0,1), (0,-1)], dtype='i,i')
    connected = False
    while not connected:
        step_i, step_j = np.random.choice(directions)
        new_i = max(min(i+step_i, grid.shape[0]-2), 1)
        new_j = max(min(j+step_j, grid.shape[1]-2), 1)
        if grid[new_i, new_j] !=1:
            i = new_i
            j = new_j
            for di, dj in directions:
                if grid[i+di, j+dj] == 1 and (np.random.rand() <= p):
                    grid[i,j] = 1
                    return grid

if __name__ == "__main__":
    L = 100
    N = 200
    grid = np.zeros((L,L), dtype="i")
    grid[L-1, L//2] = 1
    grid_2 = np.zeros((L,L), dtype="i")
    grid_2[L-1, L//2] = 1
    grid_3 = np.zeros((L,L), dtype="i")
    grid_3[L-1, L//2] = 1
    for i in tqdm(range(N)):
        grid = add_walker(grid, p=.25)
        grid_2 = add_walker(grid_2, p=.5)
        grid_3 = add_walker(grid_3, p=1)
    plt.subplot(311)
    plt.imshow(grid, cmap='binary')
    plt.title(r"p = 0.25")
    plt.subplot(312)
    plt.imshow(grid_2, cmap='binary')
    plt.title(r"p = 0.50")
    plt.subplot(313)
    plt.imshow(grid_3, cmap='binary')
    plt.title(r"p = 1")
    plt.tight_layout()
    plt.show()
