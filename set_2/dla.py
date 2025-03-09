import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool

# Input parameters
L = 100
eta = 1
omega = 1.95
delta = 1e-5
iterations = 100

# Grids

# Laplace through SOR 
# OPTIMIZE
def solve_laplace(concentration, grid, omega, delta):
    rows, cols = concentration.shape[0], concentration.shape[1]
    difference = np.inf
    while difference > delta:
        old_concentration = concentration.copy()
        for i in range(1, rows-1):
            concentration[i,0] = omega /4  * (concentration[i+1,0] + concentration[i-1,0] + concentration[i,1] + concentration[i,-2]) + (1 - omega) * concentration[i,0]
            for j in range(1, cols-1):
                if grid[i,j] != 1:
                    concentration[i, j] = omega / 4 * (concentration[i+1,j] + concentration[i-1,j] + concentration[i,j + 1] + concentration[i, j -1]) + (1 - omega) * concentration[i,j]
            concentration[i,-1] = omega /4  * (concentration[i+1,0] + concentration[i-1,0] + concentration[i,1] + concentration[i,-2]) + (1 - omega) * concentration[i,0]
        difference = np.max(np.abs(old_concentration - concentration))
    return concentration

def solve_laplace_parallel():
    pass

def grow(grid, concentration, eta):
    rows, cols = grid.shape[0], grid.shape[1]
    candidates = []
    probabilities = []
    directions = [(-1, 0), (1,0), (0,-1), (0, 1)]

    for i in range(1, rows-1):
        for j in range(1, cols - 1):
            if grid[i,j] == 0:
                for di, dj in directions:
                    if grid[i + di, j +dj] == 1:
                        candidates.append((i,j))
                        probabilities.append(concentration[i,j] ** eta)
                        break
    if np.sum(probabilities) <= 0:
        return grid, concentration
    probabilities = np.array(probabilities) / np.sum(probabilities)

    # indices of chosen location
    candidates = np.array(candidates, dtype="i,i")
    ci, cj = np.random.choice(candidates, p=probabilities)
    grid[ci, cj] = 1
    concentration[ci, cj] = 1
    return grid, concentration


# Growth Probability
if __name__ == "__main__":
    for eta, u in zip([0.3, 1, 5], [311, 312, 313]):
        grid = np.zeros((L,L), dtype=int)
        concentration = np.zeros((L,L), dtype='f')
        concentration[0] = np.ones(L)
        grid[L-1, L//2] = 1
        concentration[L-1, L//2] = 1
        for _ in tqdm(range(iterations)):
            concentration = solve_laplace(concentration, grid, omega, delta)
            grid, concentration = grow(grid, concentration, eta)
        plt.subplot(u)
        plt.imshow((grid), cmap="binary")
        plt.title(r"$\eta$= {}".format(eta))
    plt.tight_layout()
    plt.show()
