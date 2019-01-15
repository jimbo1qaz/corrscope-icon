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


gauss = lambda xs, W: np.exp(-(xs / W) ** 2)
cos_win = lambda xs: costau(xs / 4)


def win(xs):
    assert xs[0] == -1
    assert xs[-1] == 1

    W = 0.6
    e = 1
    return gauss(xs, W) * cos_win(xs) ** e


# plot
xs = np.linspace(-XMAX, XMAX, NPOINTS)


def sinusoid(dx=1, ysc=1):
    return lambda xs: sintau(xs - dx) * ysc


def plot_windowed(func, freq, **kwargs):
    plt.plot(xs, func(xs * freq) * win(xs), **kwargs)


def plot_sinusoid(dx, freq, yscale, alpha, **kwargs):
    plot_windowed(sinusoid(dx), freq=freq, alpha=alpha, label=f"{dx}", **kwargs)


NLINE = 1
max_dx = 0.2
min_alpha = 0.5


e = 0.1
for freq in np.geomspace(.4, 1.5, 4):
    plot_sinusoid(0, freq=freq, yscale=freq ** e, alpha=1)

plt.legend()
plt.gca().set_aspect("equal", adjustable="box")
plt.show()
