"""The figures' labels are matplotlib mathtext, with upright Latin particle
symbols as in the README."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from support.compare import FLOOR, checked, two_sided

# thick frames, ticks and lines, readable at the README's width
plt.rcParams.update(
    {
        "font.size": 15,  # 1.5 times matplotlib's default 10
        # (a visible box, room for the thicker dashes)
        "legend.edgecolor": "0.5",
        "legend.handlelength": 2.5,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        "xtick.minor.width": 1.2,
        "ytick.minor.width": 1.2,
        "lines.linewidth": 2.2,
        "lines.markeredgewidth": 1.5,
    }
)


IDENT = 1e-18  # where bitwise-identical values are drawn on log axes


def rel(ref, x, E=None):
    out = np.full(ref.shape, np.nan)
    sig = checked(ref, x, FLOOR, E)
    out[sig] = two_sided(ref[sig], x[sig])
    return np.where(out == 0.0, IDENT, out)
