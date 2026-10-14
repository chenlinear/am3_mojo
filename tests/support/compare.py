"""Compare checkpoint dumps of two am3 implementations.

usage: python tests/support/compare.py ref.npz test.npz [--rtol 1e-2] [--floor 1e-12]
                                        [--show 5]

Arrays are grouped by getter:
  fluxes      particle densities, component spectra, injection rates
              (every array getter except the ones below)   -> checked
  timescales  get_t_* (rates turned into times)             -> reported
  grids       get_egrid_*, scalars                          -> must be equal
The relative deviation |test/ref - 1| of every array is evaluated on its
significant bins: for fluxes, those above `floor` times the array's maximum in
particle number (the arrays are E dN/dE per ln E) or in energy (E^2 dN/dE, what
flux plots show); for timescales, those whose rate 1/t is above `floor` times
its largest. Bins far below both peaks (invisible on a flux plot, dominated by
round-off) and the longest timescales (up to the values that stand for an
absent process) carry no information. A significant bin where the test value
is not finite counts as an infinite deviation.
Arrays must have the same shape, except the two neutrino timescales that AM3
returns with the hadron grid's length (tests/reference/README.md, "Bugs of AM3
v1.2.0 that am3_mojo fixes"), compared on the common bins.
Exit status 0 if all fluxes agree to --rtol and no array is missing, differs
in shape or (grids, scalars) differs at all.
"""
import argparse, os, sys
import numpy as np

sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # tests/
from support.units import DLNE  # noqa: E402  (AM3's logarithmic grids: E_i = E_0 exp(i DLNE))

RTOL, FLOOR = 1e-2, 1e-12   # the criterion: fluxes to RTOL on the bins above FLOOR of the peak


def kind(name):
    getter = name.split('/')[-1]
    if getter.startswith('get_t_'):
        return 'timescale'
    if getter.startswith('get_egrid_'):
        return 'grid'
    return 'flux'


# getters whose array AM3 returns shorter than am3_mojo (see the docstring)
SHORTER_IN_AM3 = ('get_t_neutrino_escape', 'get_t_neutrino_extra_cooling')


def grid_of(container, key, n):
    """Energies of the n-bin array `key`: the grid of the same checkpoint
    with n bins (all of AM3's grids share the bin width, so any of them gives
    E up to a constant factor), else exp(i DLNE)."""
    pre = key.rsplit('/', 1)[0] + '/' if '/' in key else ''
    for g in container.files:
        if g.startswith(pre) and g[len(pre):].startswith('get_egrid_') and '/' not in g[len(pre):]:
            E = np.asarray(container[g], float)
            if E.shape == (n,) and np.all(E > 0):
                return E
    return np.exp(DLNE * np.arange(n))


def significant(r, floor, E=None):
    """Bins above `floor` times the peak in number (|r|) or, with the
    energies E, in energy (|E r|)."""
    a = np.abs(np.asarray(r, float))
    fin = np.isfinite(a)
    sig = np.zeros(a.shape, bool)
    if not fin.any():
        return sig
    if np.max(a[fin]) > 0:
        sig |= fin & (a > floor * np.max(a[fin]))
    if E is not None:
        e = np.where(fin, a * E, 0.0)
        if np.max(e) > 0:
            sig |= fin & (e > floor * np.max(e))
    return sig


def deviations(r, t, floor, E=None, rate=False):
    """(worst relative deviation over the significant bins, n significant) or None. With rate=True (the
    timescales), bins are significant by the rate 1/r: a timescale matters where it is short."""
    if r.size == 0 or r.ndim == 0 or r.size == 1:
        return None
    fin = np.isfinite(r)  # bins where the reference has a value
    scale = np.max(np.abs(r[fin])) if fin.any() else 0.0
    if scale == 0.0:
        # (NaN in the test compares unequal to 0 and gives inf as well)
        return (0.0 if np.max(np.abs(t[fin]), initial=0.0) == 0.0 else np.inf), 0
    if rate:
        pos = fin & (r > 0)
        sig = pos & significant(np.where(pos, 1.0 / np.where(pos, r, 1.0), 0.0), floor)
    else:
        sig = fin & significant(r, floor, E)
    with np.errstate(invalid='ignore'):
        rel = np.where(np.isfinite(t[sig]), np.abs(t[sig] / r[sig] - 1.0), np.inf)
    return rel.max(initial=0.0), int(sig.sum())


def compare(R, T, floor):
    out = dict(flux=[], timescale=[], grid=[], shape=[], missing=[])
    # every reference array must exist in the test; arrays only the test has
    # (getters of the processes am3_mojo adds) are not compared
    for k in sorted(R.files):
        if k not in T:
            out['missing'].append(k)
            continue
        r, t = R[k], T[k]
        if r.shape != t.shape:
            if k.split('/')[-1] in SHORTER_IN_AM3 and r.ndim == t.ndim == 1 and r.size < t.size:
                t = t[:r.size]
            else:
                out['shape'].append(f'{k} {r.shape} vs {t.shape}')
                continue
        kd = kind(k)
        if kd == 'grid' or r.size <= 1:
            if not np.array_equal(r, t, equal_nan=True):
                out['grid'].append((float(np.max(np.abs(t - r))), k))
            continue
        E = grid_of(R, k, r.size) if (kd == 'flux' and r.ndim == 1) else None
        d = deviations(r, t, floor, E, rate=(kd == 'timescale'))
        if d is None:
            continue
        out['flux' if kd == 'flux' else 'timescale'].append((d[0], k))
    for key in ('flux', 'timescale', 'grid'):
        out[key].sort(reverse=True)
    return out


def worst(o, key):
    """The largest deviation of compare()'s 'flux' or 'timescale' and its array, (0, '-') if there are none."""
    return o[key][0] if o[key] else (0.0, '-')


def mismatches(o):
    """compare()'s arrays that are missing in the test, differ in shape, or (grids, scalars) differ at all."""
    return o['missing'] + o['shape'] + [k for _, k in o['grid']]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('--rtol', type=float, default=RTOL)
    ap.add_argument('--floor', type=float, default=FLOOR)
    ap.add_argument('--show', type=int, default=5)
    a = ap.parse_args()
    o = compare(np.load(a.ref), np.load(a.test), a.floor)
    wf, wt = worst(o, 'flux'), worst(o, 'timescale')
    ok = wf[0] <= a.rtol and not mismatches(o)
    print(f'fluxes     : {len(o["flux"])} arrays, worst {wf[0]:.2e} ({wf[1]})  [rtol {a.rtol:g}]')
    print(f'timescales : {len(o["timescale"])} arrays, worst {wt[0]:.2e} ({wt[1]})  [reported only]')
    for d, k in o['flux'][:a.show]:
        print(f'    {d:.2e}  {k}')
    for label in ('grid', 'shape', 'missing'):
        if o[label]:
            print(f'{label} differences: {o[label][:5]}')
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
