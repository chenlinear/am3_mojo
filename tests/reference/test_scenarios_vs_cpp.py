"""am3_mojo against the C++ AM3 on the four scenarios of
tests/support/scenarios.py: every getter at every checkpoint (5,126 arrays). The
C++ has am3_mojo's fixes of AM3's bugs and its constants
(tests/reference/am3_fixes.patch), so both codes compute the same equations.

Criterion (tests/support/compare.py): fluxes agree to 1e-6 relative on every bin
above 1e-12 of the array's peak, grids exactly. Timescales are only reported.
The fluxes agree to about 1e-7, because the two codes round differently
(tests/reference/README.md, "Numerics"). A summary goes to
results/reference/summary.md after a run of all four scenarios on the default
build. The original C++ has no criterion, because am3_mojo fixes its bugs.
tests/plots/plot_spectra.py shows how far am3_mojo is from it.
"""

import os

import pytest

from support.compare import FLOOR, RTOL, compare, mismatches, worst
from support.paths import CACHE_SUFFIX, results_dir
from support.runs import cached_scenario
from support.scenarios import SCENARIOS

_rows = {}


@pytest.fixture(scope="module", autouse=True)
def summary():
    yield
    # (the kept table: only from a run of every scenario on the default build)
    if CACHE_SUFFIX or set(_rows) != set(SCENARIOS):
        return
    path = os.path.join(results_dir("reference"), "summary.md")
    with open(path, "w") as f:
        f.write(
            "| Scenario | Flux arrays | Worst flux deviation | Worst "
            "timescale deviation |\n"
        )
        f.write("|---|---|---|---|\n")
        for sc, r in _rows.items():
            f.write(f"| {sc} | {r[0]} | {r[1]:.1e} | {r[2]:.1e} |\n")


@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_fluxes_match_cpp(scenario):
    cpp = cached_scenario("cpp-fixed", scenario)
    mojo = cached_scenario("mojo-am3", scenario, rerun=True)
    o = compare(cpp, mojo, FLOOR)
    flux, ts = worst(o, "flux"), worst(o, "timescale")
    _rows[scenario] = (len(o["flux"]), flux[0], ts[0])
    assert mismatches(o) == []
    assert flux[0] <= RTOL, f"{flux[1]}: {flux[0]:.2e}"
