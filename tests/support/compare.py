"""Compare checkpoint dumps of two am3 implementations.

usage: python tests/support/compare.py ref.npz test.npz [--rtol 1e-5] [--floor 1e-12]
                                        [--show 10]

Arrays are grouped by getter:
  fluxes      particle densities, component spectra, injection rates
              (every array getter except the ones below)   -> checked
  timescales  get_t_* (rates turned into times)             -> reported
  grids       get_egrid_*, scalars                          -> must be equal
For every array the relative deviation |test/ref - 1| is evaluated on bins
above `floor` times the array's maximum (bins many orders of magnitude below
the peak carry no information on a flux plot and are dominated by round-off).
Exit status 0 if all fluxes agree to --rtol.
"""
import argparse, sys
import numpy as np


def kind(name):
    getter = name.split('/')[-1]
    if getter.startswith('get_t_'):
        return 'timescale'
    if getter.startswith('get_egrid_'):
        return 'grid'
    return 'flux'


def deviations(r, t, floor):
    """(worst relative deviation over the significant bins, n significant) or None."""
    if r.size == 0 or r.ndim == 0 or r.size == 1:
        return None
    fin = np.isfinite(r) & np.isfinite(t)
    scale = np.max(np.abs(r[fin])) if fin.any() else 0.0
    if scale == 0.0:
        return (0.0 if np.max(np.abs(t[fin]), initial=0.0) == 0.0 else np.inf), 0
    sig = fin & (np.abs(r) > floor * scale)
    rel = np.abs(t[sig] / r[sig] - 1.0)
    return rel.max(initial=0.0), int(sig.sum())


def compare(R, T, floor):
    out = dict(flux=[], timescale=[], grid=[], shape=[], missing=[])
    for k in sorted(set(R.files) | set(T.files)):
        if k not in R or k not in T:
            out['missing'].append(k)
            continue
        r, t = R[k], T[k]
        if r.shape != t.shape:
            out['shape'].append(f'{k} {r.shape} vs {t.shape}')
            continue
        kd = kind(k)
        if kd == 'grid' or r.size <= 1:
            if not np.array_equal(r, t, equal_nan=True):
                out['grid'].append((float(np.max(np.abs(t - r))), k))
            continue
        d = deviations(r, t, floor)
        if d is None:
            continue
        out['flux' if kd == 'flux' else 'timescale'].append((d[0], k))
    for key in ('flux', 'timescale', 'grid'):
        out[key].sort(reverse=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ref')
    ap.add_argument('test')
    ap.add_argument('--rtol', type=float, default=1e-5)
    ap.add_argument('--floor', type=float, default=1e-12)
    ap.add_argument('--show', type=int, default=5)
    a = ap.parse_args()
    o = compare(np.load(a.ref), np.load(a.test), a.floor)
    worst = lambda key: o[key][0] if o[key] else (0.0, '-')
    wf, wt = worst('flux'), worst('timescale')
    ok = wf[0] <= a.rtol and not o['grid'] and not o['missing']
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
