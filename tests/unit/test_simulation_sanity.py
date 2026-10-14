"""Every scenario of tests/support/scenarios.py runs, and the results are
physically sane: all fluxes finite and non-negative, the main spectra
populated, and the densities changing between checkpoints."""
import numpy as np
import pytest

from support.compare import kind
from support.runs import cached_scenario, checkpoints
from support.scenarios import SCENARIOS


@pytest.fixture(scope='module', params=sorted(SCENARIOS))
def run(request):
    return request.param, cached_scenario('mojo', request.param, rerun=True)


def test_fluxes_finite_and_non_negative(run):
    name, npz = run
    bad = [k for k in npz.files if kind(k) == 'flux' and npz[k].size > 1
           and not (np.all(np.isfinite(npz[k])) and np.all(npz[k] >= 0))]
    assert bad == [], f'{name}: {bad[:5]}'


def test_spectra_populated_and_evolving(run):
    name, npz = run
    steps = checkpoints(npz)
    assert len(steps) >= 2
    first, last = steps[0], steps[-1]
    for getter in ('get_photons', 'get_pairs'):
        assert np.max(npz[f'{last:06d}/{getter}']) > 0, f'{name}: {getter} empty'
    assert not np.array_equal(npz[f'{first:06d}/get_photons'], npz[f'{last:06d}/get_photons'])
