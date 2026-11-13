"""The top of a solver window after a sharp pulse, with am3_mojo
(tests/reference/README.md, the solver bugs "Particles that cool into a window
from far above" and "Crank-Nicolson steps at large Courant numbers").

From the second step on, inverse Compton on the synchrotron photons of the first
step cools the electrons much faster. The electrons of the first step all cool
to about the same energy within the second step. In the next step, this pulse
flows into the Chang & Cooper window, whose top moved down to about 0.1 GeV. In
the step after the pulse, the Courant number of the top cell of the window is
near 10. A Crank-Nicolson step undershoots below zero in that cell, where little
follows. So the solver's step is more implicit there. Like AM3, the solver sets
negative densities to zero after each solve, one solve per pair component. If a
component is clipped, it no longer adds up with the others to all pairs.
"""

import numpy as np
import pytest

from support.compare import significant
from support.setups import PAIR_COMPONENTS, pulse_source


def run(dt, steps):
    a = pulse_source(dt)
    E = np.array(a.get_egrid_lep())
    k = int(np.argmin(np.abs(E - 1e8)))
    rows = []
    for step in range(1, steps + 1):
        a.evolve_step()
        tot = np.array(a.get_pairs())
        parts = [np.array(getattr(a, n)()) for n in PAIR_COMPONENTS]
        sig = significant(tot, 1e-12, E) & (tot > 0)
        mismatch = np.zeros(E.size)
        mismatch[sig] = np.abs(sum(parts)[sig] / tot[sig] - 1)
        clipped = set()
        for v in parts:
            occ = np.flatnonzero(v > 0)
            if occ.size:
                clipped |= {
                    i
                    for i in range(occ[0], occ[-1] + 1)
                    if v[i] == 0.0 and sig[i] and tot[i] > 1e-6 * tot.max()
                }
        t_cool = 1.0 / (
            1.0 / a.get_t_pair_syn()[k] + 1.0 / a.get_t_pair_compton()[k]
        )
        rows.append((mismatch, int(sig.sum()), clipped, t_cool))
    return E, rows


@pytest.fixture(scope="module")
def steps_of_1e4_s():
    return run(1e4, 20)


@pytest.fixture(scope="module")
def steps_of_3000_s():
    return run(3e3, 60)


def test_cooling_after_the_first_step(steps_of_1e4_s):
    """In the second step, the synchrotron photons of the first cool the
    electrons at 0.1 GeV about 400 times faster than in the first."""
    _, rows = steps_of_1e4_s
    assert round(rows[0][3] / rows[1][3], -2) == 400


def test_nothing_clipped(steps_of_1e4_s, steps_of_3000_s):
    """At the 3000 s of test_pair_components_add_up_with_triplet and at
    1e4 s, no density goes negative. No component is zero inside its spectrum
    in any step. The components add up to all pairs to rounding (in every
    step, on at least 50 bins).
    (With Crank-Nicolson steps at any Courant number, the top cell of the
    window undershot at 1e4 s in step 4, the step after the pulse. A component
    was zero there, at about 0.1 GeV. The components added up to all pairs only
    to within 9e-2 in that bin.)"""
    for _, rows in (steps_of_1e4_s, steps_of_3000_s):
        assert min(compared for _, compared, _, _ in rows) >= 50
        assert not any(clipped for _, _, clipped, _ in rows)
        assert max(mismatch.max() for mismatch, _, _, _ in rows) < 1e-12
