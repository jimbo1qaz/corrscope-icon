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
FREQ = 1.3


def lerp(x, y, a: float):
    return x * (1 - a) + y * a


def sintau(x):
    return np.sin(tau * x)


def costau(x):
    return np.cos(tau * x)


def win(xs):
    assert xs[0] == -1
    assert xs[-1] == 1

    W = 0.5
    return np.exp(-(xs / W) ** 2) * costau(xs / 4)


# plot
xs = np.linspace(-XMAX, XMAX, NPOINTS)


def sinusoid(dx=1, ysc=1):
    return lambda xs: costau(xs - dx) * ysc


def plot_windowed(func, **kwargs):
    plt.plot(xs, func(xs * FREQ) * win(xs), **kwargs)


def plot_sinusoid(dx, alpha):
    kwargs = dict(
        #
        # color=cmap(0)
    )
    plot_windowed(sinusoid(dx=dx), alpha=alpha, label=f"{dx}", **kwargs)


NLINE = 1
max_dx = 0.4
min_alpha = 0.5

plot_sinusoid(0, 1)

for i in range(1, NLINE + 1):
    percent = i / NLINE

    dx = percent * max_dx
    alpha = lerp(1, min_alpha, percent)

    plot_sinusoid(dx=dx, alpha=alpha)
    plot_sinusoid(dx=-dx, alpha=alpha)

plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
