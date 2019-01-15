# TODO: use a real vector graphics language with gradient/shadow support

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

cmap = plt.get_cmap("tab10")


DPI = 96
width = height = 256
line_width = 4


fig = plt.gcf()
fig.set_tight_layout(False)
ax: Axes = fig.subplots(
    1,
    1,
    subplot_kw=dict(xticks=[], yticks=[]),
    gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0),
)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax.set_axis_off()
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

fig.set_dpi(DPI)
fig.set_size_inches(width / DPI, height / DPI)


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


def sinusoid(dx, freq=1, yscale=1):
    # sintau
    # x: compress=freq, shift=dx
    # y: mul=yscale
    return lambda xs: sintau((xs - dx) * freq) * yscale


def plot_sinusoid(dx, freq, yscale, alpha, color=None):
    func = sinusoid(dx, freq, yscale)
    plt.plot(xs, func(xs) * win(xs), alpha=alpha, color=color, linewidth=line_width)


NLINE = 4
max_dx = 0.2
min_alpha = 0.5


e = 0
i = NLINE - 1
for freq in np.geomspace(0.2, 1, NLINE)[::-1]:
    plot_sinusoid(0, freq=freq, yscale=freq ** e, alpha=1)
    # color=cmap(i)
    i -= 1


plt.show()
