# TODO: use a real vector graphics language with gradient/shadow support

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches

mpl.rcParams["patch.linewidth"] = 0  # doesn't work

tau = 2 * np.pi
NX = 1000
X = 1
F = 1.5


def win(xs):
    W = 0.4
    return np.exp(-(xs / W) ** 2)
    # return np.hanning(NX)


xs = np.linspace(-X, X, NX)


def graph(func):
    plt.plot(func(xs * tau * F) * win(xs))


# plt.plot(xs, ys)
graph(np.sin)
# graph(lambda xs: np.cos(xs)*.85)
graph(lambda xs: np.sin(xs + tau * 1 / 3) * 0.75)
plt.show()


# # Use LaTeX throughout the figure for consistency
# # rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 16})
# # rc('text', usetex=True)

# # Set up the figure.
# dpi = 480
# fig, ax1 = plt.subplots(
#     1, 1, figsize=(512 / dpi, 512 / dpi), dpi=dpi,
#     gridspec_kw=dict(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
# )

# SIZE = 1.1


# def setup():
#     ax1.set_xlim(-SIZE, SIZE)
#     ax1.set_ylim(-SIZE, SIZE)
#     ax1.set_axis_off()


# # The parameter t in the parametric description of the superellipse
# t = np.linspace(0, 2 * np.pi, 500)

# # Build an array of values of p up to pmax and assign the corresponding colours


# # exponent, width, height
# p = 4
# w = 1
# h = 0.9
# linewidth = 1
# color = 'k'
# bottom = '#282828'
# top = '#383838'

# text = '0CC'

# fcolor = 'w'
# font = 'clear sans'
# fy = -6
# fsize = 30
# fweight = 700
# linespacing = .9

# hi = 0.015
# hicolor = '#808080'

# shadow = -0.015
# shadow_color = 'k'

# tnames = ['', '-transparent']


# def main():
#     for transparent, suffix in enumerate(tnames):
#         ax1.cla()
#         setup()
#         # Draw superellipse
#         if not transparent:
#             kwargs = {'alpha': 1}
#             c, s = np.cos(t), np.sin(t)
#             x = np.abs(c) ** (2 / p) * np.sign(c) * w
#             y = np.abs(s) ** (2 / p) * np.sign(s) * h

#             if hi:
#                 ax1.fill(x, y + hi, c=hicolor, **kwargs)
#             if shadow:
#                 ax1.fill(x, y + shadow, c=shadow_color, **kwargs)
#             gradient_fill(x, y, bottom=bottom, top=top,
#                           ax=ax1)
#             # ax1.plot(x, y, c=color, **kwargs, linewidth=linewidth)

#         # Draw text
#         ax1.text(
#             0,
#             fy / 100,
#             text,

#             family=font,
#             fontsize=fsize, weight=fweight, linespacing=linespacing,

#             color=fcolor,
#             horizontalalignment='center',
#             verticalalignment='center',
#             zorder=100
#         )
#         plt.savefig(f'ic_launcher-web{suffix}.png', transparent=True)
#         # plt.show()


# def gradient_fill(x, y, top, bottom, ax=None, zfunc=None):
#     """based on https://stackoverflow.com/a/29347731/2683842"""
#     if ax is None:
#         ax = plt.gca()

#     # zorder = line.get_zorder() - 1
#     zorder = 10
#     alpha = 1

#     if zfunc is None:
#         h, w = 100, 1
#         z = np.empty((h, w, 4), dtype=float)
#         rgb = mcolors.colorConverter.to_rgb(top)
#         z[:, :, :3] = rgb
#         z[:, :, -1] = np.linspace(0, alpha, h)[:, None]
#     else:
#         z = zfunc(x, y, fill_color=top, alpha=alpha)
#     xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()

#     clippy = []

#     # Background fill
#     ax.fill(x, y, c=bottom, zorder=zorder)

#     # Gradient fill
#     im = ax.imshow(z, aspect='auto', extent=[xmin, xmax, ymin, ymax],
#                    origin='lower', zorder=zorder + 1)
#     clippy.append(im)

#     xy = np.column_stack([x, y])
#     clip_path = patches.Polygon(xy, facecolor='none', edgecolor='none', closed=True,
#                                 linewidth=0)
#     ax.add_patch(clip_path)

#     for clip in clippy:
#         clip.set_clip_path(clip_path)


# main()
