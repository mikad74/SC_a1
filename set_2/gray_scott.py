import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from tqdm import tqdm, trange

def grey_scott(grid, dt, dx, D_u, D_v, f, k):
    rows, cols = grid.shape[0], grid.shape[1]
    old = grid.copy()
    for i in range(1, rows-1):
        for j in range(1, cols - 1):
            grid[i,j, 0] = old[i,j, 0] + dt * (1/dx * D_u * (old[i+1,j,0] + old[i-1,j,0] + old[i,j+1,0] + old[i,j-1,0] - 4 * old[i,j,0]) - old[i,j,0] * old[i,j,1] * old[i,j,1] + f * (1 - old[i,j,0]))
            grid[i,j, 1] = old[i,j, 1] + dt * (1/dx * D_v * (old[i+1,j,1] + old[i-1,j,1] + old[i,j+1,1] + old[i,j-1,1] - 4 * old[i,j,1]) + old[i,j,0] * old[i,j,1] * old[i,j,1] - (f + k) * old[i,j,1])
    return grid


if __name__ == "__main__":
    fig, ax = plt.subplots()
    artists = []
    L = 100
    dt = 1
    dx = 1
    D_u = .16
    D_v = .08
    f = .035
    k = .06
    iterations = 2000
    v_init = 15
    grid = np.zeros((L,L, 2))
    grid[1:-1,1:-1,1] = np.random.normal(0.5, 0.02, (L-2,L-2))
    grid[:,:,0] = np.ones((L,L)) * 0.5
    grid[L//2-v_init:L//2+v_init,L//2-v_init:L//2+v_init,1] = np.ones((2*v_init, 2*v_init)) * 0.25
    for _ in trange(iterations):
        container = [ax.imshow(grid[:,:,0], animated=True)]
        grid = grey_scott(grid, dt, dx, D_u, D_v, f, k)
        artists.append(container)
    # ani = anim.ArtistAnimation(fig=fig, artists=artists, interval=5)
    # ani.save('anim.gif')
    # plt.subplot(121)
    plt.pcolormesh(grid[:,:,0], vmin=0, vmax=1)
    # plt.subplot(122)
    # plt.pcolormesh(grid[:,:,1], vmin=0, vmax=1)
    # plt.colorbar()
    plt.show()