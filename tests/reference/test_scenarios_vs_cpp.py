"""am3_mojo against the original C++ AM3 on the four simulation scenarios of
tests/support/scenarios.py: every getter at every checkpoint.

Criterion (tests/support/compare.py): fluxes (particle densities, component
spectra, injection rates) agree to 1e-5 relative on every bin above 1e-12 of
the array's peak; grids exactly. Timescales are reported only.
A summary is written to results/reference/summary.md.
"""
import os
import numpy as np
import pytest

from support.compare import compare
from support.paths import results_dir
from support.runs import cached_scenario
from support.scenarios import SCENARIOS

RTOL, FLOOR = 1e-5, 1e-12
_rows = {}


@pytest.fixture(scope='module', autouse=True)
def summary():
    yield
    path = os.path.join(results_dir('reference'), 'summary.md')
    with open(path, 'w') as f:
        f.write('| Scenario | Flux arrays | Worst flux deviation | Worst timescale deviation |\n')
        f.write('|---|---|---|---|\n')
        for sc, r in _rows.items():
            f.write(f'| {sc} | {r[0]} | {r[1]:.1e} | {r[2]:.1e} |\n')


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_fluxes_match_cpp(scenario):
    cpp = cached_scenario('cpp', scenario, rerun=True)
    mojo = cached_scenario('mojo', scenario, rerun=True)
    o = compare(cpp, mojo, FLOOR)
    worst = lambda key: o[key][0] if o[key] else (0.0, '-')
    flux, ts = worst('flux'), worst('timescale')
    _rows[scenario] = (len(o['flux']), flux[0], ts[0])
    assert not o['missing'] and not o['grid']
    assert flux[0] <= RTOL, f'{flux[1]}: {flux[0]:.2e}'
