"""am3_mojo against the C++ AM3 on the four scenarios of tests/support/scenarios.py: every getter at every checkpoint
(5,126 arrays). The C++ has am3_mojo's fixes of AM3's bugs and its constants (tests/reference/am3_fixes.patch), so both
codes compute the same equations.

Criterion (tests/support/compare.py): fluxes agree to 1e-2 relative on every bin above 1e-12 of the array's peak, grids
exactly. Timescales are only reported (on bins whose rate is above 1e-12 of its largest). Most bins agree to 1e-7; the
largest differences come from two places where the C++ result depends on rounding (tests/reference/README.md). A summary
goes to results/reference/summary.md.
"""
import os
import pytest

from support.compare import FLOOR, RTOL, compare, mismatches, worst
from support.paths import results_dir
from support.runs import cached_scenario
from support.scenarios import SCENARIOS

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
    cpp = cached_scenario('cpp-fixed', scenario, rerun=True)
    mojo = cached_scenario('mojo-am3', scenario, rerun=True)
    o = compare(cpp, mojo, FLOOR)
    flux, ts = worst(o, 'flux'), worst(o, 'timescale')
    _rows[scenario] = (len(o['flux']), flux[0], ts[0])
    assert mismatches(o) == []
    assert flux[0] <= RTOL, f'{flux[1]}: {flux[0]:.2e}'
