import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm





def solve_laplace(data):
    concentration, grid, omega, delta = data
    rows, cols = concentration.shape[0], concentration.shape[1]
    difference = np.inf
    # while difference > delta:
    old_concentration = concentration.copy()
    for i in range(1, rows-1):
        # concentration[i,0] = omega /4  * (concentration[i+1,0] + concentration[i-1,0] + concentration[i,1] + concentration[i,-2]) + (1 - omega) * concentration[i,0]
        for j in range(1, cols-1):
            if grid[i,j] != 1:
                concentration[i, j] = omega / 4 * (concentration[i+1,j] + concentration[i-1,j] + concentration[i,j + 1] + concentration[i, j -1]) + (1 - omega) * concentration[i,j]
        # concentration[i,-1] = omega /4  * (concentration[i+1,0] + concentration[i-1,0] + concentration[i,1] + concentration[i,-2]) + (1 - omega) * concentration[i,0]
    difference = np.max(np.abs(old_concentration - concentration))
    return (concentration, difference)


def solve_tester(data):
    concentration, grid, omega, delta, index= data
    rows, cols = concentration.shape[0], concentration.shape[1]
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            concentration[i,j] = index
    return (concentration, 0)

if __name__ == "__main__":
    L = 50
    delta = 1e-5
    omega = 1.3
    gridd= np.zeros((L,L), dtype="i")
    grid = np.zeros((L, L))
    grid[0] = np.zeros(L)
    grid[-1] = np.ones(L)
    N = multiprocessing.cpu_count()
    N = 2
    sub_grids = []
    diff = np.inf
    # while diff > 1e-5:
    for _ in tqdm(range(1)):
        old_grid = grid.copy()
        for i in range(N):
            idx_min, idx_max = (i*(L-2)//N,(i+1)*(L-2)//N + 2)
            if i % 2 == 0:
                sub_grids.append([grid[0:L//2 + 1, idx_min:idx_max].copy(), gridd[0:L//2 + 1, idx_min:idx_max], omega, delta])
                sub_grids.append([grid[L//2 - 1:, idx_min:idx_max].copy(), gridd[L//2 - 1:, idx_min:idx_max], omega, delta])
            else:
                sub_grids.append([grid[L//2 - 1:, idx_min:idx_max].copy(), gridd[L//2 - 1:, idx_min:idx_max], omega, delta])
                sub_grids.append([grid[0:L//2 + 1, idx_min:idx_max].copy(), gridd[0:L//2 + 1, idx_min:idx_max], omega, delta])
            list_one = [sub_grids[2*i] for i in range(len(sub_grids)//2)]
            if len(sub_grids) % 2 == 1:
                list_one.append(sub_grids[len(sub_grids)-1])
            list_two = [sub_grids[2*i+1] for i in range(len(sub_grids)//2)]
        with multiprocessing.Pool(N) as pool:
            result_one = pool.map(solve_laplace, list_one)
            pool.close()
        with multiprocessing.Pool(N) as pool:
            result_two = pool.map(solve_laplace, list_two)
            pool.close()
        

        for i in range(N):
            idx_min, idx_max = (i*(L-2)//N + 1,(i+1)*(L-2)//N + 1)
            if i % 2 == 0:
                grid[1:L//2, idx_min:idx_max] = result_one[i][0][1:-1,1:-1]
                grid[L//2:-1, idx_min:idx_max] = result_two[i][0][1:-1,1:-1]
            else:
                grid[1:L//2, idx_min:idx_max] = result_two[i][0][1:-1,1:-1]
                grid[L//2:-1, idx_min:idx_max] = result_one[i][0][1:-1,1:-1]
        for i in range(1, grid.shape[0]-1):
            grid[i,0] = omega /4  * (grid[i+1,0] + grid[i-1,0] + grid[i,1] + grid[i,-2]) + (1 - omega) * grid[i,0]
            grid[i,-1] = omega /4  * (grid[i+1,0] + grid[i-1,0] + grid[i,1] + grid[i,-2]) + (1 - omega) * grid[i,0]
        diff = np.max(np.abs(grid-old_grid))
    plt.pcolormesh(grid)
    plt.show()