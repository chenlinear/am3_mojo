"""Crank-Nicolson undershoot at the top of a solver window after a sharp pulse, with am3_mojo
(tests/reference/README.md, the solver bug "Particles that cool into a window from far above").

The set-up of tests/unit/test_added_processes.py::test_pair_components_add_up_with_triplet: electrons injected
with 1e46 erg/s into 1e48 cm^3 from 1e9 to 1e14 eV, B = 1 G, escape in 1e6 s, no hadrons. From the second step
on, inverse Compton on the synchrotron photons of the first cools the electrons up to 400 times faster (at
0.1 GeV), and the electrons of the first step all cool to about the same energy within the step. In the following
step this pulse flows into the Chang & Cooper window, whose top moved down to 0.1 GeV. Its top cell crosses about
10 cells per step, more than the 2 below which Crank-Nicolson keeps a cell's own particles positive. So in the
step after the pulse the solution there undershoots, below zero where little follows. AM3 sets negative densities
to zero after each solve, one solve per pair component, so a clipped component no longer adds up with the others
to all pairs.

At time steps of 1e4 s and the test's 3000 s, the script prints the largest |components / all pairs - 1| over
the bins the comparison criterion checks. At 1e4 s it also prints the step and bin at which a component drops to
zero inside its spectrum, and the cooling times at 0.1 and 1 GeV after steps 1 and 2. Results
(tests/reference/README.md): at 1e4 s the components add up only to 9e-2 in one bin in the step after the pulse;
at 3000 s no density goes negative.

usage: window_top_undershoot.py   (a few seconds)"""
import os, sys
import numpy as np

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))   # tests/
from support.compare import significant  # noqa: E402
from support.setups import PAIR_COMPONENTS, pulse_source  # noqa: E402


def run(dt, steps):
    """Per step: the components' mismatch, the bins where a component is zero between occupied bins, and the
    cooling times at 0.1 and 1 GeV (synchrotron and inverse Compton)."""
    a = pulse_source(dt)
    rows = []
    E = np.array(a.get_egrid_lep())
    ks = [int(np.argmin(np.abs(E - e))) for e in (1e8, 1e9)]
    for step in range(1, steps + 1):
        a.evolve_step()
        tot = np.array(a.get_pairs())
        parts = {n: np.array(getattr(a, n)()) for n in PAIR_COMPONENTS}
        comp = sum(parts.values())
        sig = significant(tot, 1e-12, E) & (tot > 0)
        mismatch = np.max(np.abs(comp[sig] / tot[sig] - 1))
        holes = []
        for n, v in parts.items():
            occ = np.nonzero(v > 0)[0]
            if occ.size:
                inside = np.arange(occ[0], occ[-1] + 1)
                holes += [(n, int(i)) for i in inside if v[i] == 0.0 and sig[i] and tot[i] > 1e-6 * tot.max()]
        t_syn, t_ic = np.array(a.get_t_pair_syn()), np.array(a.get_t_pair_compton())
        t_cool = [1.0 / (1.0 / t_syn[k] + 1.0 / t_ic[k]) for k in ks]
        rows.append((step, mismatch, holes, t_cool))
    return rows


for dt, steps in [(1e4, 20), (3e3, 60)]:
    rows = run(dt, steps)
    print(f'dt = {dt:g} s, {steps} steps: largest |components / all pairs - 1| {max(r[1] for r in rows):.1e} '
          f'(at the end {rows[-1][1]:.1e})')
    if dt == 1e4:
        for j, e in enumerate(('0.1', '1')):
            print(f'   cooling time at {e} GeV after step 1: {rows[0][3][j]:.3g} s, after step 2: {rows[1][3][j]:.3g} s '
                  f'({rows[0][3][j] / rows[1][3][j]:.0f} times faster)')
        for step, mismatch, holes, _ in rows[:8]:
            if holes:
                print(f'   step {step}: component at zero inside its spectrum: '
                      + ', '.join(f'{n[4:]} bin {i}' for n, i in holes) + f'; mismatch {mismatch:.1e}')
