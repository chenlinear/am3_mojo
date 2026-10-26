"""Particles that cool into a window of AM3's solver from far above (the solver bug of that name in
tests/reference/README.md), with am3_mojo (added processes off), the patched C++ AM3 (cpp-fixed) and the original
(cpp), each in a process of its own.

Electrons with synchrotron cooling in B = 30 G, injected per ln E as q = 1e9/E from 1e9 to 1e12 eV, in two set-ups:
  * steady state with escape (t_esc = 1e4 s): the number of electrons against q t_esc (exact: 1, as cooling keeps
    the number and no electron reaches the grid's bottom within t_esc), at dt = 0.01, 0.1, 1 and 1e4 t_esc;
  * synchrotron cooling only, from no electrons at t = 0: the number against q t (exact: 1), after 1, 2 and 5 steps
    and after all 50, 20 and 10 steps of dt = 100, 1e3 and 1e4 s.
AM3's windows take in no particles from beyond the 10 cells padded above them within a step, so where the cooling
above is faster, they lose particles. The regression tests of the fix are test_particles_cooling_into_a_window_* in
tests/unit/test_fixes.py.
"""
import json, os, subprocess, sys
import numpy as np
import pytest

from support.paths import TESTS
from support.setups import LEPTONS_ALONE, new, per_ln_e

OFF = LEPTONS_ALONE + ('quantum_syn', 'quantum_syn_cooling')
STEADY = [(1e-2, 3000), (0.1, 400), (1.0, 100), (1e4, 8)]   # time steps [t_esc] and steps of the steady state
COOLING = [(1e2, 50), (1e3, 20), (1e4, 10)]                 # time steps [s] and steps of cooling only


def electrons(impl, dt, steps, escape, checkpoints=()):
    """[(k, number of electrons / injected number - 1)] after `steps` steps and at `checkpoints`; injected: q t_esc
    with escape, q k dt without."""
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
    """The deviations of impl: {'steady': [one per time step of STEADY], 'cooling': [[(k, deviation)] per time step
    of COOLING]}."""
    return {'steady': [electrons(impl, f * 1e4, steps, True)[0][1] for f, steps in STEADY],
            'cooling': [electrons(impl, dt, steps, False, checkpoints=(1, 2, 5)) for dt, steps in COOLING]}


@pytest.fixture(scope='module')
def runs():
    """deviations() of each implementation, each in a process of its own (cpp and cpp-fixed are both the module
    am3)."""
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for impl in ('mojo-am3', 'cpp-fixed', 'cpp'):
        code = (f'import json, sys; sys.path[:0] = [{TESTS!r}, {here!r}]\n'
                f'from test_regime_boundary import deviations\nprint(json.dumps(deviations({impl!r})))')
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-3000:]
        out[impl] = json.loads(r.stdout.splitlines()[-1])
    return out


@pytest.mark.parametrize('impl', ['mojo-am3', 'cpp-fixed'])
def test_number_of_electrons_kept(runs, impl):
    """am3_mojo, and the patched C++ with the same fix, keep the number of electrons to 9e-4 in both set-ups: the
    largest |number / injected - 1| over all time steps and checkpoints rounds to 9e-4."""
    d = runs[impl]['steady'] + [x for steps in runs[impl]['cooling'] for _, x in steps]
    assert max(map(abs, d)) == pytest.approx(9e-4, abs=0.5e-4)


def test_original_am3_loses_electrons_with_cooling_only(runs):
    """With cooling only, the original AM3 has lost 3 % of the electrons after 50 steps of 100 s, 25 % after 20 steps
    of 1e3 s and 28 % after 10 steps of 1e4 s. (With escape, its other solver bugs change the number too.)"""
    loss = [-steps[-1][1] for steps in runs['cpp']['cooling']]
    assert [round(100 * x) for x in loss] == [3, 25, 28]
