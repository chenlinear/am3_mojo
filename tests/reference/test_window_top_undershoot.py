"""The Crank-Nicolson undershoot at the top of a solver window after a sharp pulse, with am3_mojo
(tests/reference/README.md, the solver bug "Particles that cool into a window from far above").

The set-up of tests/unit/test_added_processes.py::test_pair_components_add_up_with_triplet (setups.pulse_source):
electrons injected with 1e46 erg/s into 1e48 cm^3 from 1e9 to 1e14 eV, B = 1 G, escape in 1e6 s, no hadrons. From
the second step on, inverse Compton on the first step's synchrotron photons cools the electrons much faster, and the
first step's electrons all cool to about the same energy within the step. In the next step this pulse flows into the
Chang & Cooper window, whose top moved down to about 0.1 GeV. In the step after the pulse, Crank-Nicolson undershoots
in the window's top cell, below zero where little follows. AM3 sets negative densities to zero after each solve, one
per pair component, so a clipped component no longer adds up with the others to all pairs. The sums are compared on
the bins of the comparison criterion (tests/support/compare.py).
"""
import numpy as np
import pytest

from support.compare import significant
from support.setups import PAIR_COMPONENTS, pulse_source


def run(dt, steps):
    """The lepton grid and, per step: |components / all pairs - 1| in each bin the comparison criterion checks (0
    elsewhere), the bins where a component is zero between occupied bins (clipped), and the cooling time at 0.1 GeV
    (synchrotron and inverse Compton)."""
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
                clipped |= {i for i in range(occ[0], occ[-1] + 1)
                            if v[i] == 0.0 and sig[i] and tot[i] > 1e-6 * tot.max()}
        t_cool = 1.0 / (1.0 / a.get_t_pair_syn()[k] + 1.0 / a.get_t_pair_compton()[k])
        rows.append((mismatch, clipped, t_cool))
    return E, rows


@pytest.fixture(scope='module')
def steps_of_1e4_s():
    return run(1e4, 20)


@pytest.fixture(scope='module')
def steps_of_3000_s():
    return run(3e3, 60)


def test_cooling_after_the_first_step(steps_of_1e4_s):
    """In the second step, the synchrotron photons of the first cool the electrons at 0.1 GeV 400 times faster than in
    the first."""
    _, rows = steps_of_1e4_s
    assert round(rows[0][2] / rows[1][2], -2) == 400


def test_components_clipped_at_a_time_step_of_1e4_s(steps_of_1e4_s):
    """At a time step of 1e4 s, the components add up to all pairs only to within 9e-2, in one bin: in step 4, the
    step after the pulse, it is the only bin off by more than 1e-6. A component is zero there inside its spectrum, at
    about 0.1 GeV."""
    E, rows = steps_of_1e4_s
    worst = [mismatch.max() for mismatch, _, _ in rows]
    s = int(np.argmax(worst))
    mismatch, clipped, _ = rows[s]
    i = int(np.argmax(mismatch))
    assert worst[s] == pytest.approx(9e-2, abs=0.5e-2)
    assert s + 1 == 4 and np.flatnonzero(mismatch > 1e-6).tolist() == [i]
    assert i in clipped and round(E[i] / 1e9, 1) == 0.1


def test_nothing_clipped_at_3000_s(steps_of_3000_s):
    """At the 3000 s of test_pair_components_add_up_with_triplet, no density goes negative: in none of the 60 steps is
    a component zero inside its spectrum, and the components add up to all pairs to rounding."""
    _, rows = steps_of_3000_s
    assert not any(clipped for _, clipped, _ in rows)
    assert max(mismatch.max() for mismatch, _, _ in rows) < 1e-12
