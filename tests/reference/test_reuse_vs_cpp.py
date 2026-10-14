"""Reusing kernels (tests/unit/test_reuse.py) against the original C++ AM3:
am3_mojo's reused runs match the C++ reused runs (fluxes to 1e-2, see
tests/support/compare.py).

The C++ has the fixes of tests/reference/am3_fixes.patch
(tests/reference/README.md). Without them, it reads memory past the photon grid
in the hadron inverse-Compton cooling: about 1 in 10 fresh init_kernels() calls
gives garbage cooling times in the top 30 hadron bins (and wrong spectra where
protons reach them), so comparisons would fail at random.
"""
import pytest

from support.compare import FLOOR, RTOL, compare, mismatches, worst
from support.reuse_runner import CONFIGS, SETUPS
from support.runs import config, run_reuse


@pytest.fixture(scope='module')
def runs():
    r = {(s, 'cpp'): run_reuse('cpp-fixed', s, 'reused', rerun=True) for s in SETUPS}
    r.update({(s, 'mojo'): run_reuse('mojo-am3', s, 'reused', rerun=True) for s in SETUPS})
    return r


@pytest.mark.parametrize('setup', list(SETUPS))
@pytest.mark.parametrize('k', range(len(CONFIGS)))
def test_mojo_matches_cpp(runs, setup, k):
    o = compare(config(runs[setup, 'cpp'], k), config(runs[setup, 'mojo'], k), FLOOR)
    assert mismatches(o) == []
    flux = worst(o, 'flux')
    assert flux[0] <= RTOL, f'{flux[1]}: {flux[0]:.3e}'
