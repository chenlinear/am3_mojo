"""Reusing kernels (tests/unit/test_reuse.py) against the original C++ AM3.

  * the C++ original supports the same workflow: reused and fresh objects are
    bitwise identical;
  * am3_mojo's reused runs match the C++ reused runs (fluxes to 1e-5, see
    tests/support/compare.py).

The C++ used here is the original with its two out-of-bounds reads fixed
(tests/reference/oob_fix.patch). The unpatched original reads heap memory in
PionDecay::initialize_kernels for the top hadron bin; with protons at the top
of the grid (the third configuration) about 1 in 8 fresh init_kernels() calls
then gives corrupted spectra, so comparisons against it fail at random.
"""
import numpy as np
import pytest

from support.compare import compare
from support.runs import run_reuse

SETUPS = ['leptohadronic', 'blazar']
N_CONFIGS = 4
RTOL, FLOOR = 1e-5, 1e-12


@pytest.fixture(scope='module')
def runs():
    r = {(s, m): run_reuse('cpp-oobfix', s, m, rerun=True) for s in SETUPS for m in ('reused', 'fresh')}
    r.update({(s, 'mojo'): run_reuse('mojo', s, 'reused', rerun=True) for s in SETUPS})
    return r


class _Npz(dict):
    files = property(lambda self: list(self.keys()))


def getters(npz, k):
    pre = f'{k}/'
    return _Npz({n[len(pre):]: npz[n] for n in npz.files if n.startswith(pre)})


@pytest.mark.parametrize('setup', SETUPS)
@pytest.mark.parametrize('k', range(N_CONFIGS))
def test_cpp_reused_equals_fresh(runs, setup, k):
    A, B = getters(runs[setup, 'reused'], k), getters(runs[setup, 'fresh'], k)
    assert [n for n in A if not np.array_equal(A[n], B[n], equal_nan=True)] == []


@pytest.mark.parametrize('setup', SETUPS)
@pytest.mark.parametrize('k', range(N_CONFIGS))
def test_mojo_matches_cpp(runs, setup, k):
    o = compare(getters(runs[setup, 'reused'], k), getters(runs[setup, 'mojo'], k), FLOOR)
    flux = o['flux'][0]
    assert flux[0] <= RTOL, f'{flux[1]}: {flux[0]:.3e}'
