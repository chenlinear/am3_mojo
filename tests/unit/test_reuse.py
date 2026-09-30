"""Kernels initialised once can be reused for several simulations.

Workflow: AM3(); init_kernels(); then repeatedly clear_am3() -> set magnetic
field, escape/expansion timescales, pp target density, time step and
injection -> evolve, without re-initialising the kernels
(tests/support/reuse_runner.py; two process setups, four configurations, the
fourth repeating the first). Checks:
  * a reused object gives bitwise the same results as a fresh AM3 object with
    its own init_kernels() for every configuration;
  * repeating the first configuration reproduces it bitwise, so nothing leaks
    from one run into the next;
  * the configurations really produce different spectra;
  * init_kernels runs once instead of once per configuration.
"""
import numpy as np
import pytest

from support.runs import run_reuse

SETUPS = ['leptohadronic', 'blazar']
N_CONFIGS = 4


@pytest.fixture(scope='module')
def runs():
    return {(s, m): run_reuse('mojo', s, m, rerun=True) for s in SETUPS for m in ('reused', 'fresh')}


def getters(npz, k):
    pre = f'{k}/'
    return {n[len(pre):]: npz[n] for n in npz.files if n.startswith(pre)}


def bitwise_diff(A, B):
    assert A.keys() == B.keys()
    return [n for n in A if not np.array_equal(A[n], B[n], equal_nan=True)]


@pytest.mark.parametrize('setup', SETUPS)
@pytest.mark.parametrize('k', range(N_CONFIGS))
def test_reused_equals_fresh(runs, setup, k):
    assert bitwise_diff(getters(runs[setup, 'reused'], k), getters(runs[setup, 'fresh'], k)) == []


@pytest.mark.parametrize('setup', SETUPS)
def test_repeated_config_is_reproduced(runs, setup):
    npz = runs[setup, 'reused']
    assert bitwise_diff(getters(npz, 0), getters(npz, N_CONFIGS - 1)) == []


@pytest.mark.parametrize('setup', SETUPS)
def test_configs_change_the_result(runs, setup):
    npz = runs[setup, 'reused']
    spectra = [getters(npz, k)['get_photons'] for k in range(N_CONFIGS - 1)]
    for a in range(len(spectra)):
        for b in range(a + 1, len(spectra)):
            assert np.max(np.abs(spectra[a] - spectra[b])) > 1e-3 * np.max(np.abs(spectra[a]))
    assert getters(npz, 1)['get_mag_field'] == 30.0


@pytest.mark.parametrize('setup', SETUPS)
def test_reuse_skips_kernel_initialisation(runs, setup):
    assert float(runs[setup, 'reused']['__init_s']) < 0.5 * float(runs[setup, 'fresh']['__init_s'])
