from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure

cmap = get_cmap("tab10")


@dataclass
class Config:
    dim: int
    line_width: float
    nline: int


def main():
    cfgs = [
        #
        Config(256, line_width=4, nline=3),
        Config(48, line_width=2, nline=3),
        Config(32, line_width=2, nline=2),
        Config(16, line_width=1, nline=2),
    ]
    for cfg in cfgs:
        do_it(cfg)


def do_it(cfg):
    DPI = 96

    fig = Figure()
    FigureCanvasAgg(fig)
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
        ax.plot(
            xs, func(xs) * win(xs), alpha=alpha, color=color, linewidth=cfg.line_width
        )

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

    fig.savefig(f"{cfg.dim}.png", transparent=True)


if __name__ == "__main__":
    main()
