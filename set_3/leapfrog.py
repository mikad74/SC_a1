import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def F(x, k = 1):
    return -k * x

def F_ext(t, omega):
    return np.sin(omega * t)

def leapfrog(x0, k, t_max, iters, m=1, omega= 1, external=False):
    dt = t_max / iters
    history = np.zeros((iters + 1,3) )
    history[0][0] = x0
    for i in range(iters):
        x_new = history[i][0] + history[i][1] * dt
        if external:
            v_new = history[i][1] + (F_ext(i*dt, omega) + F(x_new, k))/ m * dt
        else:
            v_new = history[i][1] + F(x_new, k)/ m * dt
        history[i+1] = (x_new, v_new, i*dt)

    return history


if __name__ == "__main__":
    ks = np.array([0.1, 1, 3])
    fig, axs = plt.subplots(ks.size, 1, sharex=True)
    for i, k in enumerate(ks):
        results = leapfrog(1, k, 10, 1000)
        x = results[:,0]
        v = results[:,1]
        t = results[:, 2]
        ax1 = axs[i]
        ax1.plot(t, x, color='black', label=f"k = {k}")
        ax1.set_ylabel("x(t)")
        ax1.tick_params(axis='y', labelcolor='black')
        ax2 = ax1.twinx()
        ax2.plot(t, v, color='red')
        ax2.set_ylabel("v(t)")
        ax2.tick_params(axis='y', labelcolor='red')
        ax1.legend()
    axs[-1].set_xlabel("T(t)")
    plt.show()
    
    omegas = [.1, 1, 10]
    for omega in omegas:
        results_2 = leapfrog(1, 1, 15, 1000, omega=omega, external=True)
        x = results_2[:,0]
        v = results_2[:,1]
        plt.plot(x, v, label=r"$\omega$ = {}".format(omega))
    plt.title("Phase plot of harmonic oscillator with external driving force, k=1")
    plt.xlabel("x(t)")
    plt.ylabel("v(t)")
    plt.legend()
    plt.show()
    

    

