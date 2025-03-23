import numpy as np
import matplotlib.pyplot as plt



N = 4
x = np.linspace(0, 4, N, endpoint=False)
y = np.linspace(0, 4, N, endpoint=False)
for i in range(N):
    for j in range(N):
        plt.plot(i, j, 'ko')
        plt.text(i+0.1, j, f'{i + j * N}', ha='left', va='center', fontsize=12)

plt.xlim((-.5, 3.5))
plt.grid()
plt.show()