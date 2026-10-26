"""Kernels initialised once can be reused for several simulations (tests/support/reuse_runner.py: init_kernels()
once, then per configuration clear_am3(), set the parameters, evolve). Two process setups, four configurations, the
fourth repeating the first. Checks:
  * from the second configuration on, a reused object gives bitwise the results of a fresh one with its own
    init_kernels() (in the first, both modes run the same calls);
  * the repeated first configuration is reproduced bitwise: nothing leaks from one run into the next.
"""
import numpy as np
import pytest

from support.reuse_runner import CONFIGS, SETUPS
from support.runs import config, run_reuse


@pytest.fixture(scope='module')
def runs():
    return {(s, m): run_reuse('mojo', s, m, rerun=True) for s in SETUPS for m in ('reused', 'fresh')}


def bitwise_diff(A, B):
    assert A.keys() == B.keys()
    return [n for n in A if not np.array_equal(A[n], B[n], equal_nan=True)]


@pytest.mark.parametrize('setup', list(SETUPS))
@pytest.mark.parametrize('k', range(1, len(CONFIGS)))
def test_reused_equals_fresh(runs, setup, k):
    assert bitwise_diff(config(runs[setup, 'reused'], k), config(runs[setup, 'fresh'], k)) == []


@pytest.mark.parametrize('setup', list(SETUPS))
def test_repeated_config_is_reproduced(runs, setup):
    npz = runs[setup, 'reused']
    assert bitwise_diff(config(npz, 0), config(npz, len(CONFIGS) - 1)) == []
