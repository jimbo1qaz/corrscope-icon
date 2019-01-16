import subprocess
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import get_cmap
from matplotlib.figure import Figure
from matplotlib.image import AxesImage

cmap = get_cmap("tab10")
order = [0, 3, 2, 1]
cmap = [cmap(x) for x in order].__getitem__


@dataclass
class Config:
    dim: int
    line_width: float
    nline: int


def main():
    cfgs = [
        #
        Config(256, line_width=3, nline=2),
        Config(96, line_width=2, nline=2),
        Config(48, line_width=1.5, nline=2),
        Config(32, line_width=1, nline=2),
        Config(16, line_width=0.75, nline=2),
    ]
    fnames = []
    for cfg in cfgs:
        ret = do_it(cfg)
        fnames.append(ret.fname)

    subprocess.run(["magick", "convert"] + fnames + ["icon.ico"], check=True)


@dataclass
class FigAx:
    fig: Figure
    ax: Axes


@dataclass
class Ret:
    fname: str


def do_it(cfg) -> Ret:
    DPI = 96
    FLOAT = np.float32
    GAMMA = 2.2

    def get_fig_ax() -> FigAx:
        fig = Figure()
        fig.set_facecolor("#00000000")
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

        return FigAx(fig, ax)

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

    def plot_sinusoid(dx, freq, yscale, color=None):
        func = sinusoid(dx, freq, yscale)
        ys = func(xs) * win(xs)

        line_plot.ax.plot(xs, ys, color=color, linewidth=cfg.line_width)

        fill = get_fig_ax()
        fill.ax.fill_between(xs, ys, 0, facecolor=color)
        fill_plots.append(fill)

    line_plot: FigAx = get_fig_ax()
    fill_plots: List[FigAx] = []

    RGB_A = Tuple[np.ndarray, np.ndarray]
    RGBA = np.ndarray

    def get_rgb_a(rgba):
        return rgba[:-1], rgba[-1:]

    def render_rgba(plot) -> np.ndarray:
        """ returns int """
        canvas: FigureCanvasAgg = plot.fig.canvas
        canvas.draw()

        wh = canvas.get_width_height()
        rgba = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(*wh, 4)
        return rgba

    def get_premul_planar_rgba(plot: FigAx) -> RGBA:
        rgba = render_rgba(plot)
        planar_rgba = np.moveaxis(rgba, -1, 0).astype(FLOAT, order="C")
        planar_rgba /= 255

        rgb, a = get_rgb_a(planar_rgba)
        # linearize
        rgb **= GAMMA
        rgb *= a
        return planar_rgba

    def compute_image(fill_alpha) -> Figure:
        """
        my idea is "draw each fill individually opaque",
        "sum up premultiplied rgba values",
        "divide rgb by sum of alpha",
        "replace alpha with max alpha of any fill", and
        "draw lines on top"
        """

        rgba_s = [get_premul_planar_rgba(plot) for plot in fill_plots]
        # rgbs = [rgba[:, :-1] for rgba in rgbas]
        # alphas = [rgba[:, -1:] for rgba in rgbas]

        premul_planar_rgba = np.sum(rgba_s, 0)
        rgb, a = get_rgb_a(premul_planar_rgba)

        # clip saturated image regions (still premultiplied)
        premul_planar_rgba /= np.maximum(a, 1)

        # Transform premul to regular
        assert rgb.any()
        assert np.amax(a) == 1
        rgb /= np.maximum(a, 1e-6)  # epsilon
        assert np.amax(rgb) <= 1, np.amax(rgb)

        # compress to gamma
        rgb **= 1 / GAMMA

        # Overlay color
        a *= fill_alpha

        # uwu
        rgba = np.moveaxis(premul_planar_rgba, 0, -1).copy("C")
        assert rgba.dtype == np.float32
        del premul_planar_rgba, rgb, a

        img: AxesImage = line_plot.ax.imshow(rgba, extent=[-1, 1, -1, 1])
        # img.set_zorder(-100)

        # return render_rgba(line_plot)
        return line_plot.fig

    top = "narrow"
    blue = "narrow"

    top_blue = top == blue
    if top_blue:
        i = cfg.nline - 1
        di = -1
    else:
        i = 0
        di = 1

    freqs = np.geomspace(0.3, 1, cfg.nline)
    if top == "wide":
        freqs = freqs[::-1]

    e = 0

    for freq in freqs:
        plot_sinusoid(0, freq=freq, yscale=freq ** e, color=cmap(i))
        i += di

    fname = f"{cfg.dim}.png"
    compute_image(fill_alpha=0.5).savefig(fname, transparent=True)
    return Ret(fname)


if __name__ == "__main__":
    main()
