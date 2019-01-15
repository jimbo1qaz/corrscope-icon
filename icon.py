# TODO: use a real vector graphics language with gradient/shadow support

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches


cmap = plt.get_cmap("tab10")


mpl.rcParams["patch.linewidth"] = 0  # doesn't work


# params
tau = 2 * np.pi
NPOINTS = 1000
XMAX = 1
FREQ = 1.5


def NOT(x):
    return 1 - x


def sintau(x):
    return np.sin(tau * x)


def costau(x):
    return np.cos(tau * x)


def win(xs):
    W = 0.4
    return np.exp(-(xs / W) ** 2)
    # return np.hanning(NPOINTS)


# plot
xs = np.linspace(-XMAX, XMAX, NPOINTS)


def sinusoid(dx=1, ysc=1):
    return lambda xs: costau(xs - dx) * ysc


def plot_windowed(func, **kwargs):
    plt.plot(xs, func(xs * FREQ) * win(xs), **kwargs)


def plot_sinusoid(dx, alpha):
    plot_windowed(sinusoid(dx=dx), alpha=alpha, label=f"{dx}")
    # color=cmap(0)


NLINE = 2
den_dx = NLINE
den_alpha = NLINE + 1

for i in range(1, NLINE + 1)[::-1]:
    dx = i / den_dx
    alpha = NOT(i / den_alpha) ** 3
    plot_sinusoid(dx=dx, alpha=alpha)
    plot_sinusoid(dx=-dx, alpha=alpha)

plot_sinusoid(0, 1)

plt.legend()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
