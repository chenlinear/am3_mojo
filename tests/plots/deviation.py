"""The style of the figures of tests/plots/ (their rcParams) and their per-bin deviation, on the bins the comparisons
check (tests/support/compare.py: above FLOOR of the peak in number or energy). The figures' labels are matplotlib
mathtext, with upright Latin particle symbols as in the README."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({  # thick frames, ticks and lines, readable at the README's width
    'font.size': 15,  # 1.5 times matplotlib's default 10
    'legend.edgecolor': '0.5', 'legend.handlelength': 2.5,  # (a visible box, room for the thicker dashes)
    'axes.linewidth': 1.5, 'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'xtick.minor.width': 1.2, 'ytick.minor.width': 1.2, 'lines.linewidth': 2.2, 'lines.markeredgewidth': 1.5})

from support.compare import FLOOR, significant

IDENT = 1e-18   # where bitwise-identical values are drawn on log axes


def rel(ref, x, E=None):
    """|x/ref - 1| on the significant bins of ref (with the energies E: also in energy), NaN elsewhere; IDENT
    where the two are equal."""
    out = np.full(ref.shape, np.nan)
    if not np.any(ref):
        return out
    sig = significant(ref, FLOOR, E)
    out[sig] = np.abs(x[sig] / ref[sig] - 1.0)
    return np.where(out == 0.0, IDENT, out)
