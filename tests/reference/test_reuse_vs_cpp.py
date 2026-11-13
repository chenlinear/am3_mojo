"""Reusing kernels (tests/unit/test_reuse.py) against the patched C++ AM3:
am3_mojo's reused runs match the C++ ones (fluxes to 1e-6,
tests/support/compare.py).

The C++ has the fixes of tests/reference/am3_fixes.patch. Without them, its
hadron inverse-Compton cooling reads memory past the photon grid. About 1 in 10
fresh init_kernels() calls then gives garbage cooling times in the top 30 hadron
bins (and wrong spectra where protons reach them). Comparisons then fail at
random.
"""

import pytest

from support.compare import FLOOR, RTOL, compare, mismatches, worst
from support.reuse_runner import CONFIGS, SETUPS
from support.runs import config, run_reuse


@pytest.fixture(scope="module")
def runs():
    return {
        (s, code): run_reuse(impl, s, "reused", rerun=impl != "cpp-fixed")
        for code, impl in (("cpp", "cpp-fixed"), ("mojo", "mojo-am3"))
        for s in SETUPS
    }


@pytest.mark.parametrize("setup", list(SETUPS))
@pytest.mark.parametrize("k", range(len(CONFIGS)))
def test_mojo_matches_cpp(runs, setup, k):
    o = compare(
        config(runs[setup, "cpp"], k), config(runs[setup, "mojo"], k), FLOOR
    )
    assert mismatches(o) == []
    flux = worst(o, "flux")
    assert flux[0] <= RTOL, f"{flux[1]}: {flux[0]:.3e}"
