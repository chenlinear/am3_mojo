"""Particles that cool into a window of AM3's solver from far above (the solver bug of that name in
tests/reference/README.md), with am3_mojo (added processes off) or the C++ AM3 builds of `make reference`.

Electrons with synchrotron cooling in B = 30 G, injected per ln E as q = 1e9/E between 1e9 and 1e12 eV:
  * steady state with escape (t_esc = 1e4 s): the number of electrons against q t_esc (exact: 1, as cooling
    keeps the number and no electron reaches the grid's bottom within t_esc), at dt = 0.01, 0.1, 1 and 1e4 t_esc;
  * synchrotron cooling only, from no electrons at t = 0: the number against q t (exact: 1), at dt = 100, 1e3
    and 1e4 s.
Within a step, AM3's windows took in no particles from beyond the 10 cells padded above them. Where the cooling
above is faster, it lost 3 %, 23 % and 11 % of the electrons in the steady state with the other fixes applied,
and 3 % and 25-28 % with cooling only. am3_mojo and the patched C++ (cpp-fixed) count them, to 8e-4 in the steady
state and with cooling only. The original AM3 (cpp) has other solver bugs too. --threshold sets threshold_matrix
(default 10, an e-fold): the solver uses Chang & Cooper (regime 2) where particles cool by fewer cells per step.

usage: regime_boundary.py [--impl mojo-am3|cpp|cpp-fixed] [--threshold T]   (a few seconds)"""
import argparse, os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.output import quiet  # noqa: E402
from support.setups import new, per_ln_e  # noqa: E402

OFF = ['hadronic', 'electron_compton', 'annihilation', 'ssa', 'proton_syn', 'expansion', 'adiabatic_cooling',
       'quantum_syn', 'quantum_syn_cooling']


def electrons(impl, dt, steps, escape, threshold, checkpoints=()):
    """[(k, number of electrons / injected number - 1)] after `steps` steps and at `checkpoints`; injected: q t_esc
    with escape, q k dt without."""
    a = new(impl, OFF, dt, init=False, escape=escape, electron_syn=True)
    out = []
    if threshold is not None:
        a.set_solver_threshold_matrix(threshold)
    quiet(a.init_kernels)
    a.set_mag_field(30.0)
    a.set_escape_timescale(1e4)
    E = np.array(a.get_egrid_lep())
    q = per_ln_e(E)
    a.set_injection_rate_electrons(list(q))
    for k in range(1, steps + 1):
        a.evolve_step()
        if k in checkpoints or k == steps:
            n = np.sum(np.array(a.get_electrons()))
            out.append((k, n / (np.sum(q) * (1e4 if escape else k * dt)) - 1))
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--impl', default='mojo-am3')
    ap.add_argument('--threshold', type=float, default=None)
    a = ap.parse_args()
    thr = 'default (10)' if a.threshold is None else f'{a.threshold:g}'
    print(f'{a.impl}, threshold_matrix {thr}')
    print('steady state with escape, number / (q t_esc) - 1:')
    for f, steps in [(1e-2, 3000), (0.1, 400), (1.0, 100), (1e4, 8)]:
        (_, d), = electrons(a.impl, f * 1e4, steps, True, a.threshold)
        print(f'  dt = {f:g} t_esc: {d:+.2e}')
    print('cooling only, from no electrons, number / (q t) - 1 after k steps:')
    for dt, steps in [(1e2, 50), (1e3, 20), (1e4, 10)]:
        r = electrons(a.impl, dt, steps, False, a.threshold, checkpoints=(1, 2, 5))
        print(f'  dt = {dt:g} s: ' + ', '.join(f'{k}: {d:+.2e}' for k, d in r))
