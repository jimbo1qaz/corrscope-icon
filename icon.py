from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

cmap = plt.get_cmap("tab10")


@dataclass
class Config:
    dim: int
    line_width: float
    nline: int


cfg = Config(256, 4, 4)

DPI = 96


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
fig.set_size_inches(cfg.dim / DPI, cfg.dim / DPI)


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
    plt.plot(xs, func(xs) * win(xs), alpha=alpha, color=color, linewidth=cfg.line_width)


top = "narrow"
blue = "narrow"


top_blue = top == blue
if top_blue:
    i = cfg.nline - 1
    di = -1
else:
    i = 0
    di = 1

freqs = np.geomspace(0.2, 1, cfg.nline)
if top == "wide":
    freqs = freqs[::-1]

e = 0

for freq in freqs:
    plot_sinusoid(0, freq=freq, yscale=freq ** e, alpha=1, color=cmap(i))
    i += di


plt.savefig(f'{cfg.dim}.png', transparent=True)
