"""Particles that cool into a window of AM3's solver from far above (the solver
bug of that name in tests/reference/README.md). Electrons with synchrotron
cooling, in two set-ups:
  * Steady state with escape: the number of electrons against q t_esc
    (exact: 1, because cooling keeps the number and no electron reaches the
    grid's bottom within t_esc).
  * Synchrotron cooling only, from no electrons at t = 0: the number against
    q t (exact: 1).
AM3's windows take in no particles from beyond the 10 cells padded above them
within a step. So where the cooling above is faster, the windows lose particles.
The regression tests of the fix are test_particles_cooling_into_a_window_* in
tests/unit/test_fixes.py.
"""

import json, os, subprocess, sys

import numpy as np
import pytest

from support.paths import TESTS
from support.setups import LEPTONS_ALONE, new, per_ln_e

OFF = LEPTONS_ALONE + ("quantum_syn", "quantum_syn_cooling")
# time steps [t_esc] and steps
STEADY = [(1e-2, 3000), (0.1, 400), (1.0, 100), (1e4, 8)]
# time steps [s] and steps
COOLING = [(1e2, 50), (1e3, 20), (1e4, 10)]


def electrons(impl, dt, steps, escape, checkpoints=()):
    a = new(impl, OFF, dt, escape=escape, electron_syn=True)
    out = []
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


def deviations(impl):
    return {
        "steady": [
            electrons(impl, f * 1e4, steps, True)[0][1] for f, steps in STEADY
        ],
        "cooling": [
            electrons(impl, dt, steps, False, checkpoints=(1, 2, 5))
            for dt, steps in COOLING
        ],
    }


@pytest.fixture(scope="module")
def runs():
    """cpp and cpp-fixed are both the module am3, so each implementation runs in
    a process of its own."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for impl in ("mojo-am3", "cpp-fixed", "cpp"):
        code = (
            f"import json, sys; sys.path[:0] = [{TESTS!r}, {here!r}]\n"
            "from test_regime_boundary import deviations\n"
            f"print(json.dumps(deviations({impl!r})))"
        )
        r = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-3000:]
        out[impl] = json.loads(r.stdout.splitlines()[-1])
    return out


@pytest.mark.parametrize("impl", ["mojo-am3", "cpp-fixed"])
def test_number_of_electrons_kept(runs, impl):
    """am3_mojo, and the patched C++ with the same fix, keep the number of
    electrons to 9e-4 in both set-ups. The largest |number / injected - 1| over
    all time steps and checkpoints rounds to 9e-4."""
    d = runs[impl]["steady"] + [
        x for steps in runs[impl]["cooling"] for _, x in steps
    ]
    assert max(map(abs, d)) == pytest.approx(9e-4, abs=0.5e-4)


def test_original_am3_loses_electrons_with_cooling_only(runs):
    """With cooling only, the original AM3 loses 3 % of the electrons in
    50 steps of 100 s, 25 % in 20 steps of 1e3 s and 28 % in 10 steps of 1e4 s.
    (With escape, its other solver bugs change the number too.)"""
    loss = [-steps[-1][1] for steps in runs["cpp"]["cooling"]]
    assert [round(100 * x) for x in loss] == [3, 25, 28]
