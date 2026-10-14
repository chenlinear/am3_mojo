"""Triplet pair production, README "Triplet pair production": the extrapolation of the product
distributions above the last table node (k = 1e10), where the Monte Carlo of tools/triplet runs out
of double precision. physics/triplet.mojo fits two asymptotic laws to the nodes k >= 1e6 (K_FIT):
  at fixed y = x k <= 100 (Y_SW):   dN/dln y = g0(y) + g1(y) / ln k
  at fixed x = y / k >= 1e-5 (X_LO): k dN/dln y = A(x) ln k + B(x)
and continues the second below X_LO as (a0 ln y + b0) / y. Here the code (through
tests/harness/triplet_mojo.mojo) makes the same fits to fewer nodes, and its extrapolation is compared
with the nodes left out (one to three decades in k): the L1 distance of the normalised distributions,
the same weighted with the product energy, and the ratio of the mean energy fractions.
usage: extrapolation.py"""
import os, sys
import numpy as np
sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))   # tests/
from support.harness import Triplet  # noqa: E402

ROLES = ('positron', 'lower-energy electron', 'recoil (xi)')
T = Triplet()
LY0, DLY = T.ly0, T.dly


def compare(fit_lo, fit_hi, targets):
    nodes = [n for n, k in enumerate(T.k) if fit_lo * 0.999 <= k <= fit_hi * 1.001]
    print(f'fit to the {len(nodes)} nodes {fit_lo:.0e} <= k <= {fit_hi:.0e}:')
    left_out = [int(np.argmin(np.abs(np.log(T.k / kt)))) for kt in targets]   # the node nearest to each target
    extrapolated = T.extrapolated(fit_lo * 0.999, fit_hi * 1.001, T.k[left_out])
    for n, ex in zip(left_out, extrapolated):
        k = T.k[n]
        ly = LY0 + (np.arange(ex.shape[1]) + 0.5) * DLY
        y = np.exp(ly)
        cols = []
        for r in range(3):
            tr = np.zeros(ex.shape[1])
            d = T.dist[n][r]
            tr[:min(len(d), len(tr))] = d[:len(tr)]
            tr /= tr.sum() * DLY
            l1 = np.sum(np.abs(ex[r] - tr)) * DLY
            l1e = np.sum(np.abs(ex[r] - tr) * y) / np.sum(tr * y)
            mr = np.sum(ex[r] * y) / np.sum(tr * y)
            cols.append(f'{ROLES[r]}: L1 {l1:.3f}, energy-weighted {l1e:.3f}, mean ratio {mr:.4f}')
        print(f'  k = {k:.3g} ({np.log10(k / fit_hi):.0f} decade(s) beyond the fit): ' + ';  '.join(cols))


if __name__ == '__main__':
    compare(1e6, 1e9, [1e10])
    compare(1e6, 1e8, [1e9, 1e10])
    compare(1e5, 1e7, [1e8, 1e9, 1e10])
    print('the code fits the nodes 1e6 <= k <= 1e10 and uses the laws above 1e10')
