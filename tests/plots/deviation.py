"""The per-bin deviation the figures of tests/plots/ draw: on the bins the comparisons check
(tests/support/compare.py: above FLOOR of the peak in number or energy)."""
import numpy as np

from support.compare import FLOOR, significant

IDENT = 1e-18   # where bitwise-identical values are drawn on log axes


def rel(ref, x, E=None):
    """|x/ref - 1| on the significant bins of ref (with the energies E: also in energy), NaN elsewhere; IDENT
    where the two are equal."""
    out = np.full(ref.shape, np.nan)
    if ref.size == 0 or not np.any(ref != 0):
        return out
    sig = significant(ref, FLOOR, E)
    out[sig] = np.abs(x[sig] / ref[sig] - 1.0)
    return np.where(out == 0.0, IDENT, out)
